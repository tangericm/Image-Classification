import time
import streamlit as st
import matplotlib.pyplot as plt
import os
from train import TrainConfig, train_model, MODEL_REGISTRY

def format_time(seconds: float) -> str:
    """Format seconds into H:MM:SS."""
    # Convert to integers
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60

    parts = []
    if h > 0:
        parts.append(f"{h} h")
    if m > 0:
        parts.append(f"{m} m")
    parts.append(f"{s} s")

    return " ".join(parts)
    

def main():
    st.set_page_config(page_title="CIFAR-10 Image Classification", layout="wide")
    st.title("CIFAR-10 Image Classification")

    st.markdown(
        "Deep learning image classification training on the CIFAR-10 Dataset https://www.cs.toronto.edu/~kriz/cifar.html.  \n"
        "CIFAR-10: 50,000 training images and 10,000 test images in 10 classes. "
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    model_name = st.sidebar.selectbox(
        "Model",
        options=list(MODEL_REGISTRY.keys()),
        index=0,  # default to AlexNet
    )

    N = st.sidebar.number_input(
        "Number of Images for Training", min_value=0, max_value=50000, value=50000, step=1000
    )

    num_epochs = st.sidebar.slider("Epochs", min_value=1, max_value=100, value=20)
    batch_size = st.sidebar.selectbox(
        "Batch size", options=[64, 128, 256, 512], index=2
    )
    learning_rate = st.sidebar.number_input(
        "Learning rate", min_value=1e-6, max_value=1e-1, value=1e-4, step=1e-5, format="%.5f"
    )
    device = st.sidebar.selectbox("Device", options=["cuda", "cpu"], index=0)

    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            st.sidebar.warning("CUDA not available; falling back to CPU")
            device = "cpu"

    st.sidebar.markdown("---")
    st.sidebar.markdown("Click **Train** to start.")

    # Main interface
    if st.button("Train"):
        config = TrainConfig(
            model_name=model_name,
            input_shape=(3, 64, 64),
            num_classes=10,
            N=N,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=42,
            device=device,
        )

        st.write("### Running training...")
        st.write("Configuration:")
        st.json({k: str(v) for k, v in config.__dict__.items()})

        # Placeholders for progress and plots
        status_text = st.empty()
        progress_bar = st.progress(0)

        # Side-by-side plots
        col_loss, col_acc = st.columns(2)
        loss_plot_placeholder = col_loss.empty()
        acc_plot_placeholder = col_acc.empty()

        start_time = time.perf_counter()

        def on_epoch(epoch_idx, history):
            """Callback from trainer: update progress, ETA, and plots."""
            completed = epoch_idx + 1
            frac = completed / config.num_epochs
            progress_bar.progress(frac)

            elapsed = time.perf_counter() - start_time
            avg_epoch = elapsed / completed
            remaining = avg_epoch * (config.num_epochs - completed)

            elapsed_str = format_time(elapsed)
            remaining_str = format_time(remaining)
            status_text.markdown(
                f"**Epoch {completed}/{config.num_epochs}**  \n"
                f"Time Elapsed: `{elapsed_str}`  \n"
                f"Estimated Time Remaining: `{remaining_str}`"
            )

            epochs = list(range(1, completed + 1))

            # ---- Loss plot ----
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(epochs, history.train_loss, label="Train Loss")
            ax_loss.plot(epochs, history.val_loss, label="Val Loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(True)
            loss_plot_placeholder.pyplot(fig_loss)
            plt.close(fig_loss)

            # ---- Accuracy plot ----
            fig_acc, ax_acc = plt.subplots()
            ax_acc.plot(epochs, history.train_acc, label="Train Acc")
            ax_acc.plot(epochs, history.val_acc, label="Val Acc")
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy (%)")
            ax_acc.set_ylim(0, 100)
            ax_acc.legend()
            ax_acc.grid(True)
            acc_plot_placeholder.pyplot(fig_acc)

            # If this is the final epoch, save the final plots to the checkpoint dir
            try:
                if completed == config.num_epochs:
                    ckpt_dir = getattr(history, "ckpt_dir", None)
                    if ckpt_dir:
                        os.makedirs(ckpt_dir, exist_ok=True)

                        # Save loss figure
                        loss_png = os.path.join(ckpt_dir, "loss.png")
                        fig_loss.savefig(loss_png)

                        # Save accuracy figure
                        acc_png = os.path.join(ckpt_dir, "accuracy.png")
                        fig_acc.savefig(acc_png)
            except Exception as e:
                try:
                    st.warning(f"Failed to save final plots: {e}")
                except Exception:
                    pass

            plt.close(fig_acc)

        # Run training with streaming updates
        history = train_model(config, epoch_callback=on_epoch)

        st.markdown("### Run Summary")
        st.write(f"Run name: `{history.run_name}`")
        st.write(f"Metrics CSV: `{history.metrics_file}`")
        st.write(f"Checkpoints directory: `{history.ckpt_dir}`")

        # Final status
        total_time = time.perf_counter() - start_time
        progress_bar.progress(1.0)
        status_text.markdown(
            f"**Training complete** in `{format_time(total_time)}`  \n"
            f"Final Test Accuracy: **{history.test_acc:.2f}%**"
        )


if __name__ == "__main__":
    main()
