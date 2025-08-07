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
        false,
        |selected| async move {
            if selected.is_empty() {
                return Ok(());
            }
            
            // Show confirmation
            println!("\nSelected containers to {}:", action);
            for container in &selected {
                println!("  - {} ({}) - {}", container.display_name(), container.short_id(), container.status.display_string());
            }
            
            // Always force remove all selected containers
            let removable_containers = selected;
            
            print!("\nAre you sure you want to {} these containers? [y/N]: ", action);
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            if input.trim().to_lowercase() == "y" || input.trim().to_lowercase() == "yes" {
                let mut success_count = 0;
                let mut failed_count = 0;
                
                for container in removable_containers {
                    print!("{}ing container {} ... ", if force { "Force remov" } else { "Remov" }, container.display_name());
                    io::stdout().flush()?;
                    
                    let mut args = vec!["rm", "-f"];
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

pub async fn status_command() -> Result<()> {
    // First, get and display the formatted table output
    let table_output = tokio::process::Command::new("docker")
        .args(&["stats", "--no-stream"])
        .output()
        .await;

    match table_output {
        Ok(output) if output.status.success() => {
            let table_data = String::from_utf8_lossy(&output.stdout);
            println!("{}", table_data);
        }
        _ => {
            println!("No running containers found.");
            return Ok(());
        }
    }

    // Then, get CSV format for parsing statistics
    let csv_output = tokio::process::Command::new("docker")
        .args(&["stats", "--no-stream", "--format", "{{.Container}},{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.NetIO}},{{.BlockIO}},{{.PIDs}}"])
        .output()
        .await;

    match csv_output {
        Ok(output) if output.status.success() => {
            let stats_output = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = stats_output.lines().filter(|line| !line.is_empty()).collect();
            
            if lines.is_empty() {
                return Ok(());
            }
            
            // Calculate totals from all CSV lines
            let mut total_cpu = 0.0;
            let mut total_mem_usage = 0;
            let mut total_net_in = 0;
            let mut total_net_out = 0;
            let mut total_block_in = 0;
            let mut total_block_out = 0;
            
            for line in lines {
                // CSV format: CONTAINER,NAME,CPU%,MEM USAGE / LIMIT,NET I/O,BLOCK I/O,PIDS
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 6 {
                    // Parse CPU percentage (remove % sign) - parts[2]
                    if let Some(cpu_str) = parts[2].strip_suffix('%') {
                        if let Ok(cpu) = cpu_str.parse::<f64>() {
                            total_cpu += cpu;
                        }
                    }
                    
                    // Parse memory usage (format: "123.4MiB / 1.234GiB") - parts[3]
                    if let Some(mem_part) = parts[3].split(" / ").next() {
                        total_mem_usage += parse_memory_size(mem_part);
                    }
                    
                    // Parse network I/O (format: "1.23kB / 4.56MB") - parts[4]
                    let net_parts: Vec<&str> = parts[4].split(" / ").collect();
                    if net_parts.len() == 2 {
                        total_net_in += parse_byte_size(net_parts[0]);
                        total_net_out += parse_byte_size(net_parts[1]);
                    }
                    
                    // Parse block I/O (format: "1.23MB / 4.56GB") - parts[5]
                    let block_parts: Vec<&str> = parts[5].split(" / ").collect();
                    if block_parts.len() == 2 {
                        total_block_in += parse_byte_size(block_parts[0]);
                        total_block_out += parse_byte_size(block_parts[1]);
                    }
                }
            }
            
            // Print totals
            println!("\n📊 Total Statistics:");
            println!("CPU: {:.2}%", total_cpu);
            println!("MEM USAGE: {}", format_byte_size(total_mem_usage));
            println!("NET I/O: {} / {}", format_byte_size(total_net_in), format_byte_size(total_net_out));
            println!("BLOCK I/O: {} / {}", format_byte_size(total_block_in), format_byte_size(total_block_out));
        }
        Ok(output) => {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            eprintln!("Failed to get container statistics: {}", error_msg.trim());
        }
        Err(e) => {
            eprintln!("Error executing docker stats: {}", e);
        }
    }
    
    Ok(())
}

// Helper function to parse memory size (e.g., "123.4MiB" -> bytes)
fn parse_memory_size(size_str: &str) -> u64 {
    let size_str = size_str.trim();
    if let Some(pos) = size_str.find(|c: char| c.is_alphabetic()) {
        let (number_part, unit_part) = size_str.split_at(pos);
        if let Ok(number) = number_part.parse::<f64>() {
            let multiplier = match unit_part.to_uppercase().as_str() {
                "B" => 1,
                "KIB" | "KB" => 1_024,
                "MIB" | "MB" => 1_024 * 1_024,
                "GIB" | "GB" => 1_024 * 1_024 * 1_024,
                "TIB" | "TB" => 1_024_u64.pow(4),
                _ => 1,
            };
            return (number * multiplier as f64) as u64;
        }
    }
    0
}

// Helper function to parse byte size (e.g., "1.23kB" -> bytes)
fn parse_byte_size(size_str: &str) -> u64 {
    let size_str = size_str.trim();
    if let Some(pos) = size_str.find(|c: char| c.is_alphabetic()) {
        let (number_part, unit_part) = size_str.split_at(pos);
        if let Ok(number) = number_part.parse::<f64>() {
            let multiplier = match unit_part.to_uppercase().as_str() {
                "B" => 1,
                "KB" => 1_000,
                "MB" => 1_000_000,
                "GB" => 1_000_000_000,
                "TB" => 1_000_000_000_000,
                "KIB" => 1_024,
                "MIB" => 1_024 * 1_024,
                "GIB" => 1_024 * 1_024 * 1_024,
                "TIB" => 1_024_u64.pow(4),
                _ => 1,
            };
            return (number * multiplier as f64) as u64;
        }
    }
    0
}

// Helper function to format bytes to human readable format
fn format_byte_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    const THRESHOLD: f64 = 1000.0;
    
    if bytes == 0 {
        return "0B".to_string();
    }
    
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= THRESHOLD && unit_index < UNITS.len() - 1 {
        size /= THRESHOLD;
        unit_index += 1;
    }
    
    if size >= 100.0 {
        format!("{:.0}{}", size, UNITS[unit_index])
    } else if size >= 10.0 {
        format!("{:.1}{}", size, UNITS[unit_index])
    } else {
        format!("{:.2}{}", size, UNITS[unit_index])
    }
}