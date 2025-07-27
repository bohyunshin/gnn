import argparse
import copy
import os
import time
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from gnn.libs.parse_args import parse_args
from gnn.libs.config import load_yaml
from gnn.libs.utils import get_model_module, accuracy
from gnn.libs.logger import setup_logger
from gnn.libs.plot import plot_metric_at_k
from gnn.data.data_loader import DataLoader
from gnn.data.data_splitter import train_val_test_split
from gnn.preprocess.preprocessor import preprocess_adjacency_matrix


ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT_PATH, "./data/{data_name}")
DATA_CONFIG_PATH = os.path.join(ROOT_PATH, "./config/data.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")


def main(args: argparse.ArgumentParser) -> None:
    # setup result path
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test = "test" if args.is_test else "untest"
    result_path = RESULT_PATH.format(test=test, model=args.model_name, dt=dt)
    os.makedirs(result_path, exist_ok=True)
    logger = setup_logger(os.path.join(result_path, "log.log"))

    logger.info(f"model: {args.model_name}")
    if args.model_name == "graphsage":
        logger.info(f"graphsage aggregator function: {args.sage_aggregator}")
    elif args.model_name == "gat":
        logger.info(f"number of heads when multi-head attention: {args.num_heads}")
        logger.info(f"alpha in leaky relu: {args.leaky_relu_alpha}")
    elif args.model_name == "fastgcn":
        logger.info(f"sample size: {args.fastgcn_sample_size}")
        logger.info(f"selected sampling method: {args.fastgcn_sampling_method}")
    logger.info(f"learning rate: {args.learning_rate}")
    logger.info(f"weight decay: {args.weight_decay}")
    logger.info(f"dropout: {args.dropout}")
    logger.info(f"epochs: {args.epochs}")
    logger.info(f"test mode: {args.is_test}")

    # load data config
    data_config = load_yaml(DATA_CONFIG_PATH).get(args.data_name)

    # load data
    data_loader = DataLoader(
        data_path=DATA_PATH.format(data_name=args.data_name),
        data_name=args.data_name,
        logger=logger,
    )
    features, adj, labels, idx_map = data_loader.load()

    # preprocess adjacency matrix depending on selected model
    adj = preprocess_adjacency_matrix(
        adj=adj,
        model_name=args.model_name,
    )

    # get tr / val / test index for semi-supervised learning
    idx_train, idx_val, idx_test = train_val_test_split(
        num_train=data_config.num_train,
        num_val=data_config.num_val,
        n=labels.shape[0],
    )

    logger.info(f"Number of total nodes: {labels.shape[0]}")
    logger.info(
        f"Number of train nodes: {len(idx_train)}, {round(len(idx_train) / labels.shape[0], 4)} out of total"
    )
    logger.info(
        f"Number of val nodes: {len(idx_val)}, {round(len(idx_val) / labels.shape[0], 4)} out of total"
    )
    logger.info(
        f"Number of test nodes: {len(idx_test)}, {round(len(idx_test) / labels.shape[0], 4)} out of total"
    )

    # model and optimizer
    model_module = get_model_module(model_name=args.model_name)
    model = model_module(
        num_feature=features.shape[1],
        hidden_dim=features.shape[1] // 2,
        num_class=labels.max().item() + 1,
        dropout=args.dropout,
        aggregator=args.sage_aggregator,
        num_heads=args.num_heads,
        alpha=args.leaky_relu_alpha,
        sample_size=args.fastgcn_sample_size,
        sampling_method=args.fastgcn_sampling_method,
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_start_time = time.time()
    best_loss = float("inf")
    tr_losses = []
    tr_accs = []
    val_losses = []
    val_accs = []
    early_stopping = False
    for epoch in range(args.epochs):
        logger.info(f"################## epoch {epoch} ##################")
        epoch_start_time = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        tr_loss = F.nll_loss(output[idx_train], labels[idx_train])
        tr_acc = accuracy(output[idx_train], labels[idx_train])
        tr_loss.backward()
        optimizer.step()

        val_loss = F.nll_loss(output[idx_val], labels[idx_val])
        val_acc = accuracy(output[idx_val], labels[idx_val])

        tr_loss = tr_loss.item()
        tr_acc = tr_acc.item()
        val_loss = val_loss.item()
        val_acc = val_acc.item()
        tr_losses.append(tr_loss)
        tr_accs.append(tr_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.info(f"Train loss: {round(tr_loss, 4)}")
        logger.info(f"Train accuracy: {round(tr_acc, 4)}")
        logger.info(f"Val loss: {round(val_loss, 4)}")
        logger.info(f"Val accuracy: {round(val_acc, 4)}")
        logger.info(
            f"Elapsed time for current epoch: {round(time.time() - epoch_start_time, 4)}"
        )

        # early stopping logic
        if best_loss > val_loss:
            prev_best_loss = best_loss
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience = args.patience
            torch.save(
                model.state_dict(),
                os.path.join(result_path, "weight.pt"),
            )
            logger.info(
                f"Best validation: {round(best_loss, 4)}, Previous validation loss: {round(prev_best_loss, 4)}"
            )
        else:
            patience -= 1
            logger.info(f"Validation loss did not decrease. Patience {patience} left.")
            if patience == 0:
                logger.info(
                    f"Patience over. Early stopping at epoch {epoch} with {round(best_loss, 4)} validation loss"
                )
                early_stopping = True

        if early_stopping is True:
            break

    logger.info("Optimization Finished!")
    logger.info(f"Total time elapsed: {round(time.time() - train_start_time, 4)}")

    # test metric calculation
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    logger.info(f"Test loss: {round(loss_test.item(), 4)}")
    logger.info(f"Test accuracy: {round(acc_test.item(), 4)}")

    # Load the best model weights
    model.load_state_dict(best_model_weights)
    logger.info("Load weight with best validation loss")

    # torch.save(model.state_dict(), os.path.join(result_path, "weight.pt"))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "num_feature": features.shape[1],
                "hidden_dim": features.shape[1] // 2,
                "num_class": labels.max().item() + 1,
                "dropout": args.dropout,
            },
            "features": features,
            "adj": adj,
            "idx_map": idx_map,
        },
        os.path.join(result_path, "train_result.pt"),
    )

    # summarize training results
    plot_metric_at_k(
        tr_loss=tr_losses,
        tr_acc=tr_accs,
        val_loss=val_losses,
        val_acc=val_accs,
        parent_save_path=result_path,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
