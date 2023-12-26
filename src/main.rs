pub mod inference;
pub mod training;

use clap::{Parser, Subcommand};

use crate::inference::ui;
use crate::training::run;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// 训练
    Training,
    /// 推理
    Inference,
}
fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Training) => {
            run::run();
        }
        Some(Commands::Inference) => {
            ui::ui().unwrap();
        }
        None => {
            println!("这是一个命令行工具，请使用 --help 查看帮助");
        }
    }
}
