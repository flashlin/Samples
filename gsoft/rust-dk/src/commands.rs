use anyhow::Result;
use std::io::{self, Write};

use crate::docker::DockerClient;
use crate::ui::{display_container_table, run_interactive_selection};

pub async fn show_container_status() -> Result<()> {
    let docker = DockerClient::new()?;
    let containers = docker.list_containers(true).await?;
    
    if containers.is_empty() {
        println!("No containers found.");
        return Ok(());
    }
    
    // Separate running and stopped containers
    let running: Vec<_> = containers.iter().filter(|c| c.status.is_running()).collect();
    let stopped: Vec<_> = containers.iter().filter(|c| !c.status.is_running()).collect();
    
    if !running.is_empty() {
        println!("🟢 Running Containers:");
        display_container_table(&running.into_iter().cloned().collect::<Vec<_>>())?;
    }
    
    if !stopped.is_empty() {
        println!("🔴 Stopped Containers:");
        display_container_table(&stopped.into_iter().cloned().collect::<Vec<_>>())?;
    }
    
    Ok(())
}

pub async fn logs_command() -> Result<()> {
    let docker = DockerClient::new()?;
    let containers = docker.list_containers(true).await?;
    
    if containers.is_empty() {
        println!("No containers found.");
        return Ok(());
    }
    
    run_interactive_selection(
        containers,
        "Select a container to view logs",
        false,
        |selected| async move {
            if let Some(container) = selected.first() {
                println!("Viewing logs for container: {}", container.display_name());
                
                // 使用 tokio::process::Command 直接執行
                let status = tokio::process::Command::new("docker")
                    .args(&["logs", "-f", &container.id])
                    .status()
                    .await;
                
                match status {
                    Ok(exit_status) if exit_status.success() => {
                        println!("日誌查看完成");
                    }
                    Ok(_) => {
                        eprintln!("Docker logs 命令執行失敗");
                    }
                    Err(e) => {
                        eprintln!("執行 docker logs 時發生錯誤: {}", e);
                    }
                }
            }
            Ok(())
        },
    ).await?;
    
    Ok(())
}

pub async fn bash_command() -> Result<()> {
    let docker = DockerClient::new()?;
    let containers = docker.list_containers(true).await?;
    
    if containers.is_empty() {
        println!("No containers found.");
        return Ok(());
    }
    
    run_interactive_selection(
        containers,
        "Select a container to enter bash/shell",
        false,
        |selected| async move {
            if let Some(container) = selected.first() {
                println!("Entering shell for container: {}", container.display_name());
                
                // 先嘗試 bash
                let bash_status = tokio::process::Command::new("docker")
                    .args(&["exec", "-it", &container.id, "bash"])
                    .status()
                    .await;
                
                match bash_status {
                    Ok(exit_status) if exit_status.success() => {
                        println!("Shell 會話結束");
                        return Ok(());
                    }
                    _ => {
                        // bash 失敗，嘗試 sh
                        println!("bash 不可用，嘗試 sh...");
                        let sh_status = tokio::process::Command::new("docker")
                            .args(&["exec", "-it", &container.id, "sh"])
                            .status()
                            .await;
                        
                        match sh_status {
                            Ok(exit_status) if exit_status.success() => {
                                println!("Shell 會話結束");
                            }
                            Ok(_) => {
                                eprintln!("無法進入容器 shell");
                            }
                            Err(e) => {
                                eprintln!("執行 docker exec 時發生錯誤: {}", e);
                            }
                        }
                    }
                }
            }
            Ok(())
        },
    ).await?;
    
    Ok(())
}

pub async fn rm_command(force: bool) -> Result<()> {
    let docker = DockerClient::new()?;
    let containers = docker.list_containers(true).await?;
    
    if containers.is_empty() {
        println!("No containers found.");
        return Ok(());
    }
    
    let action = if force { "force remove" } else { "remove" };
    
    run_interactive_selection(
        containers,
        &format!("Select containers to {}", action),
        true,
        |selected| async move {
            if selected.is_empty() {
                return Ok(());
            }
            
            // Show confirmation
            println!("\nSelected containers to {}:", action);
            for container in &selected {
                println!("  - {} ({})", container.display_name(), container.short_id());
            }
            
            print!("\nAre you sure you want to {} these containers? [y/N]: ", action);
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            if input.trim().to_lowercase() == "y" || input.trim().to_lowercase() == "yes" {
                for container in selected {
                    print!("{}ing container {} ... ", if force { "Force remov" } else { "Remov" }, container.display_name());
                    io::stdout().flush()?;
                    
                    let mut args = vec!["rm"];
                    if force {
                        args.push("-f");
                    }
                    args.push(&container.id);
                    
                    let status = tokio::process::Command::new("docker")
                        .args(&args)
                        .status()
                        .await;
                    
                    match status {
                        Ok(exit_status) if exit_status.success() => {
                            println!("✓ Done");
                        }
                        Ok(_) => {
                            println!("✗ Failed");
                        }
                        Err(e) => {
                            println!("✗ Error: {}", e);
                        }
                    }
                }
            } else {
                println!("Operation cancelled.");
            }
            
            Ok(())
        },
    ).await?;
    
    Ok(())
}

pub async fn restart_command() -> Result<()> {
    let docker = DockerClient::new()?;
    let containers = docker.list_containers(true).await?;
    
    if containers.is_empty() {
        println!("No containers found.");
        return Ok(());
    }
    
    run_interactive_selection(
        containers,
        "Select containers to restart",
        true,
        |selected| async move {
            if selected.is_empty() {
                return Ok(());
            }
            
            // Show confirmation
            println!("\nSelected containers to restart:");
            for container in &selected {
                println!("  - {} ({})", container.display_name(), container.short_id());
            }
            
            print!("\nAre you sure you want to restart these containers? [y/N]: ");
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            if input.trim().to_lowercase() == "y" || input.trim().to_lowercase() == "yes" {
                for container in selected {
                    print!("Restarting container {} ... ", container.display_name());
                    io::stdout().flush()?;
                    
                    let status = tokio::process::Command::new("docker")
                        .args(&["restart", &container.id])
                        .status()
                        .await;
                    
                    match status {
                        Ok(exit_status) if exit_status.success() => {
                            println!("✓ Done");
                        }
                        Ok(_) => {
                            println!("✗ Failed");
                        }
                        Err(e) => {
                            println!("✗ Error: {}", e);
                        }
                    }
                }
            } else {
                println!("Operation cancelled.");
            }
            
            Ok(())
        },
    ).await?;
    
    Ok(())
}