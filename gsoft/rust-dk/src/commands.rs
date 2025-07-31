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
                println!("  - {} ({}) - {}", container.display_name(), container.short_id(), container.status.display_string());
            }
            
            // Check for running containers when not using force
            if !force {
                let running_containers: Vec<_> = selected.iter()
                    .filter(|c| c.status.is_running())
                    .collect();
                
                if !running_containers.is_empty() {
                    println!("\n⚠️  Warning: The following containers are running:");
                    for container in &running_containers {
                        println!("  - {} ({})", container.display_name(), container.short_id());
                    }
                    println!("Running containers cannot be removed without force. Use 'dk rm -f' to force remove.");
                    return Ok(());
                }
            }
            
            print!("\nAre you sure you want to {} these containers? [y/N]: ", action);
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            if input.trim().to_lowercase() == "y" || input.trim().to_lowercase() == "yes" {
                let mut success_count = 0;
                let mut failed_count = 0;
                
                for container in selected {
                    print!("{}ing container {} ... ", if force { "Force remov" } else { "Remov" }, container.display_name());
                    io::stdout().flush()?;
                    
                    let mut args = vec!["rm"];
                    if force {
                        args.push("-f");
                    }
                    args.push(&container.id);
                    
                    let output = tokio::process::Command::new("docker")
                        .args(&args)
                        .output()
                        .await;
                    
                    match output {
                        Ok(output) if output.status.success() => {
                            println!("✓ Done");
                            success_count += 1;
                        }
                        Ok(output) => {
                            let error_msg = String::from_utf8_lossy(&output.stderr);
                            println!("✗ Failed: {}", error_msg.trim());
                            failed_count += 1;
                        }
                        Err(e) => {
                            println!("✗ Error: {}", e);
                            failed_count += 1;
                        }
                    }
                }
                
                println!("\nSummary: {} containers removed successfully, {} failed.", success_count, failed_count);
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

pub async fn clean_command() -> Result<()> {
    println!("🧹 Starting Docker cleanup...\n");
    
    // Clean up dangling images (<none>)
    println!("1. Cleaning up dangling images (<none>)...");
    let dangling_output = tokio::process::Command::new("docker")
        .args(&["images", "-f", "dangling=true", "-q"])
        .output()
        .await;
    
    match dangling_output {
        Ok(output) if output.status.success() => {
            let dangling_ids = String::from_utf8_lossy(&output.stdout);
            let ids: Vec<&str> = dangling_ids.lines().filter(|id| !id.is_empty()).collect();
            
            if ids.is_empty() {
                println!("   ✓ No dangling images found");
            } else {
                println!("   Found {} dangling images", ids.len());
                
                // Show confirmation for dangling images
                print!("   Remove these dangling images? [y/N]: ");
                io::stdout().flush()?;
                
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                
                if input.trim().to_lowercase() == "y" || input.trim().to_lowercase() == "yes" {
                    let mut success_count = 0;
                    for id in ids {
                        print!("   Removing {} ... ", id);
                        io::stdout().flush()?;
                        
                        let rm_output = tokio::process::Command::new("docker")
                            .args(&["rmi", id])
                            .output()
                            .await;
                        
                        match rm_output {
                            Ok(rm_output) if rm_output.status.success() => {
                                println!("✓ Done");
                                success_count += 1;
                            }
                            Ok(rm_output) => {
                                let error_msg = String::from_utf8_lossy(&rm_output.stderr);
                                println!("✗ Failed: {}", error_msg.trim());
                            }
                            Err(e) => {
                                println!("✗ Error: {}", e);
                            }
                        }
                    }
                    println!("   ✓ Removed {} dangling images", success_count);
                } else {
                    println!("   Skipped dangling images removal");
                }
            }
        }
        Ok(output) => {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            println!("   ✗ Failed to list dangling images: {}", error_msg.trim());
        }
        Err(e) => {
            println!("   ✗ Error listing dangling images: {}", e);
        }
    }
    
    // Clean up build cache older than 2 months
    println!("\n2. Cleaning up build cache older than 2 months...");
    let builder_output = tokio::process::Command::new("docker")
        .args(&["builder", "prune", "-f", "--filter", "until=2m"])
        .output()
        .await;
    
    match builder_output {
        Ok(output) if output.status.success() => {
            let result = String::from_utf8_lossy(&output.stdout);
            println!("   ✓ Build cache cleanup completed");
            println!("   {}", result.trim());
        }
        Ok(output) => {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            println!("   ✗ Failed to clean build cache: {}", error_msg.trim());
        }
        Err(e) => {
            println!("   ✗ Error cleaning build cache: {}", e);
        }
    }
    
    // Additional cleanup: system prune
    println!("\n3. Running system prune...");
    let system_output = tokio::process::Command::new("docker")
        .args(&["system", "prune", "-f"])
        .output()
        .await;
    
    match system_output {
        Ok(output) if output.status.success() => {
            let result = String::from_utf8_lossy(&output.stdout);
            println!("   ✓ System cleanup completed");
            println!("   {}", result.trim());
        }
        Ok(output) => {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            println!("   ✗ Failed to run system prune: {}", error_msg.trim());
        }
        Err(e) => {
            println!("   ✗ Error running system prune: {}", e);
        }
    }
    
    println!("\n🎉 Docker cleanup completed!");
    Ok(())
}