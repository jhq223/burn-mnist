## 这是一个使用rust编写的，带有gui的mnist识别器。

### 项目结构
```
📦mnist-burn
 ┣ 📂src
 ┃ ┣ 📂inference
 ┃ ┃ ┣ 📜image.rs
 ┃ ┃ ┣ 📜inference.rs
 ┃ ┃ ┣ 📜mod.rs
 ┃ ┃ ┣ 📜state.rs
 ┃ ┃ ┗ 📜ui.rs
 ┃ ┣ 📂training
 ┃ ┃ ┣ 📜data.rs
 ┃ ┃ ┣ 📜mod.rs
 ┃ ┃ ┣ 📜model.rs
 ┃ ┃ ┣ 📜run.rs
 ┃ ┃ ┗ 📜training.rs
 ┃ ┗ 📜main.rs
 ┣ 📜Cargo.lock
 ┣ 📜Cargo.toml
 ┗ 📜README.md
```

### 快速开始
从下载并在命令行运行


### 自行编译
从GitHub克隆
```bash
git clone https://github.com/jhq223/burn-mnist.git
```

```bash
cd burn-mnist
cargo build --release
```

