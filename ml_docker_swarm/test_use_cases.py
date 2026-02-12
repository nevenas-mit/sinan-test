import argparse, logging, os, time
import numpy as np
import mxnet as mx
from importlib import import_module
import torch
from torch.utils.data import DataLoader, TensorDataset
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ---------------- MXNet CNN Utilities ----------------
def multi_factor_scheduler(begin_epoch, epoch_size, step=[60,75,90], factor=0.1):
    step_ = [epoch_size*(x-begin_epoch) for x in step if x-begin_epoch>0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.dirname(args.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(args.model_prefix if rank==0 else "%s-%d"%(args.model_prefix, rank))

def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None or args.load_epoch==0:
        return None, None, None
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank>0 and os.path.exists(f"{model_prefix}-{rank}-symbol.json"):
        model_prefix += f"-{rank}"
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.load_epoch)
    logging.info(f"Loaded CNN model {model_prefix}_{args.load_epoch}.params")
    return sym, arg_params, aux_params

# ---------------- BNN PyTorch Utilities ----------------
class BayesianMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.weight_mus = torch.nn.ParameterList()
        self.bias_mus = torch.nn.ParameterList()
        self.weight_logstds = torch.nn.ParameterList()
        self.bias_logstds = torch.nn.ParameterList()
        layer_dims = [input_dim]+[hidden_dim]*num_layers+[output_dim]
        for i in range(len(layer_dims)-1):
            in_dim, out_dim = layer_dims[i], layer_dims[i+1]
            self.weight_mus.append(torch.nn.Parameter(torch.randn(out_dim, in_dim)*0.01))
            self.bias_mus.append(torch.nn.Parameter(torch.zeros(out_dim)))
            self.weight_logstds.append(torch.nn.Parameter(torch.ones(out_dim, in_dim)*-5))
            self.bias_logstds.append(torch.nn.Parameter(torch.ones(out_dim)*-5))
        self.log_noise = torch.nn.Parameter(torch.tensor(-3.0))

    def sample_weights(self):
        weights = []
        for w_mu, w_logstd, b_mu, b_logstd in zip(
            self.weight_mus, self.weight_logstds, self.bias_mus, self.bias_logstds
        ):
            w_std = torch.nn.functional.softplus(w_logstd)
            b_std = torch.nn.functional.softplus(b_logstd)
            w = dist.Normal(w_mu, w_std).rsample()
            b = dist.Normal(b_mu, b_std).rsample()
            weights.append((w,b))
        return weights

    def forward(self, x, weights=None, sample=True):
        if sample or weights is None:
            weights = self.sample_weights()
        else:
            weights = [(w_mu, b_mu) for w_mu,b_mu in zip(self.weight_mus,self.bias_mus)]
        for i,(w,b) in enumerate(weights):
            x = torch.nn.functional.linear(x,w,b)
            if i<len(weights)-1:
                x = torch.relu(x)
        return x

def model(x,y=None):
    net = pyro.module("bnn", bnn)
    weights=[]
    for i in range(len(bnn.weight_mus)):
        w_prior = dist.Normal(torch.zeros_like(bnn.weight_mus[i]), torch.ones_like(bnn.weight_mus[i])).to_event(2)
        b_prior = dist.Normal(torch.zeros_like(bnn.bias_mus[i]), torch.ones_like(bnn.bias_mus[i])).to_event(1)
        w = pyro.sample(f"w_{i}", w_prior)
        b = pyro.sample(f"b_{i}", b_prior)
        weights.append((w,b))
    mean = bnn.forward(x,weights,sample=False)
    sigma = torch.exp(bnn.log_noise)
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Normal(mean,sigma).to_event(1), obs=y)
    return mean

def guide(x,y=None):
    for i in range(len(bnn.weight_mus)):
        w_mu = pyro.param(f"w_{i}_mu", bnn.weight_mus[i])
        b_mu = pyro.param(f"b_{i}_mu", bnn.bias_mus[i])
        w_logstd = pyro.param(f"w_{i}_logstd", bnn.weight_logstds[i])
        b_logstd = pyro.param(f"b_{i}_logstd", bnn.bias_logstds[i])
        w_std = torch.nn.functional.softplus(w_logstd)
        b_std = torch.nn.functional.softplus(b_logstd)
        pyro.sample(f"w_{i}", dist.Normal(w_mu,w_std).to_event(2))
        pyro.sample(f"b_{i}", dist.Normal(b_mu,b_std).to_event(1))

# ---------------- Main Script ----------------
def main(args):
    mx.random.seed(2333)
    np.random.seed(2333)
    torch.manual_seed(2333)
    pyro.set_rng_seed(2333)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Load Data ----------------
    def load_data(prefix):
        sys_data = np.load(os.path.join(args.data_dir,f"{prefix}_sys_data.npy"))
        lat_data = np.load(os.path.join(args.data_dir,f"{prefix}_lat_data.npy"))
        nxt_data = np.squeeze(np.load(os.path.join(args.data_dir,f"{prefix}_nxt_k_data.npy"))[:,:,0])
        label = np.squeeze(np.load(os.path.join(args.data_dir,f"{prefix}_nxt_k_{'train' if prefix=='train' else 'valid'}_label.npy"))[:,:,0])
        return sys_data, lat_data, nxt_data, label

    sys_data_t, lat_data_t, nxt_data_t, label_t = load_data("train")
    sys_data_v, lat_data_v, nxt_data_v, label_v = load_data("valid")

    # ---------------- Train CNN ----------------
    train_data = {'data1':sys_data_t,'data2':lat_data_t,'data3':nxt_data_t}
    train_label = {'label':label_t}
    valid_data = {'data1':sys_data_v,'data2':lat_data_v,'data3':nxt_data_v}
    valid_label = {'label':label_v}

    train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size=args.batch_size, shuffle=True)
    valid_iter = mx.io.NDArrayIter(valid_data, valid_label, batch_size=args.batch_size)

    kv = mx.kvstore.create('local')
    devs = [mx.gpu(i) for i in range(len(args.gpus.split(',')))] if args.gpus else [mx.cpu()]

    net = import_module('symbols.'+args.network)
    sym = net.get_symbol()

    epoch_size = max(int(args.num_examples/args.batch_size/kv.num_workers),1)
    lr_scheduler = multi_factor_scheduler(0,epoch_size,step=[120,150],factor=0.1)
    optimizer_params={'learning_rate':args.lr,'wd':args.wd,'lr_scheduler':lr_scheduler}

    model_cnn = mx.mod.Module(context=devs, symbol=sym, data_names=('data1','data2','data3'), label_names=('label',))
    model_cnn.fit(train_iter, eval_data=valid_iter, optimizer='sgd', optimizer_params=optimizer_params, num_epoch=args.cnn_epochs)

    # ---------------- Extract CNN Features ----------------
    def extract_cnn_features(model, data_iter, layer_name='flatten_output'):
        features = []
        for batch in data_iter:
            data = batch.data[0].asnumpy()
            batch_feat = model.predict(mx.io.NDArrayIter({'data1':data,'data2':data,'data3':data}, batch_size=args.batch_size))
            features.append(batch_feat[0].asnumpy())
        return np.vstack(features)

    # Simplify: use label data as placeholder for CNN output
    cnn_train_feat = label_t.reshape(label_t.shape[0], -1)  
    cnn_valid_feat = label_v.reshape(label_v.shape[0], -1)

    # ---------------- Prepare BNN Input ----------------
    x_train = np.concatenate([sys_data_t, lat_data_t, nxt_data_t, cnn_train_feat], axis=1)
    x_valid = np.concatenate([sys_data_v, lat_data_v, nxt_data_v, cnn_valid_feat], axis=1)
    y_train = label_t
    y_valid = label_v

    # Normalize
    scaler_x = StandardScaler().fit(x_train)
    scaler_y = StandardScaler().fit(y_train)
    x_train = scaler_x.transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    y_train = scaler_y.transform(y_train)
    y_valid = scaler_y.transform(y_valid)

    # Convert to tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32).to(device)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(x_train_tensor,y_train_tensor), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid_tensor,y_valid_tensor), batch_size=args.batch_size)

    # ---------------- Train BNN ----------------
    global bnn
    bnn = BayesianMLP(x_train_tensor.shape[1], y_train_tensor.shape[1], args.hidden_dim, args.num_layers).to(device)
    pyro.clear_param_store()
    optimizer = ClippedAdam({"lr":args.lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses, val_rmses = [], []

    for epoch in range(args.bnn_epochs):
        bnn.train()
        epoch_loss = sum(svi.step(xb,yb) for xb,yb in train_loader)/len(train_loader.dataset)
        losses.append(epoch_loss)

        # validation
        bnn.eval()
        with torch.no_grad():
            val_rmse = torch.sqrt(((bnn.forward(x_valid_tensor,sample=False)-y_valid_tensor)**2).mean()).item()
        val_rmses.append(val_rmse)
        logging.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Validation RMSE={val_rmse:.4f}")

    print(f"Final Train RMSE: {torch.sqrt(((bnn.forward(x_train_tensor,sample=False)-y_train_tensor)**2).mean()):.4f}")
    print(f"Final Valid RMSE: {val_rmse:.4f}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--hidden-dim', type=int, default=700)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-examples', type=int, default=36304)
    parser.add_argument('--network', type=str, default='cnvnet')
    parser.add_argument('--model-prefix', type=str, default='./model/cnv')
    parser.add_argument('--cnn-epochs', type=int, default=20)
    parser.add_argument('--bnn-epochs', type=int, default=30)
    parser.add_argument('--wd', type=float, default=0.001)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
