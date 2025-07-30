use clap::{Parser, Subcommand};
use anyhow::Result;

mod docker;
mod ui;
mod commands;
mod types;

use commands::*;

#[derive(Parser)]
#[command(name = "dk")]
#[command(about = "A friendly and interactive Docker container management tool")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// View container logs interactively
    Logs,
    /// Enter container bash/shell interactively
    Bash,
    /// Remove containers interactively
    Rm {
        /// Force remove containers
        #[arg(short, long)]
        force: bool,
    },
    /// Restart containers interactively
    Restart,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        None => {
            // Default behavior: show container status
            show_container_status().await?;
        }
        Some(Commands::Logs) => {
            logs_command().await?;
        }
        Some(Commands::Bash) => {
            bash_command().await?;
        }
        Some(Commands::Rm { force }) => {
            rm_command(force).await?;
        }
        Some(Commands::Restart) => {
            restart_command().await?;
        }
    }
    
    Ok(())
}