use anyhow::{Result, Context};
use bollard::Docker;
use bollard::container::ListContainersOptions;
use bollard::models::ContainerSummary;
use chrono::{DateTime, Utc};

use crate::types::{ContainerInfo, ContainerStatus};

pub struct DockerClient {
    client: Docker,
}

impl DockerClient {
    pub fn new() -> Result<Self> {
        let client = Docker::connect_with_local_defaults()
            .context("Failed to connect to Docker daemon. Make sure Docker is running.")?;
        
        Ok(Self { client })
    }
    
    pub async fn list_containers(&self, all: bool) -> Result<Vec<ContainerInfo>> {
        let options = ListContainersOptions::<String> {
            all,
            ..Default::default()
        };
        
        let containers = self.client
            .list_containers(Some(options))
            .await
            .context("Failed to list containers")?;
        
        let mut container_infos = Vec::new();
        
        for container in containers {
            if let Some(info) = self.convert_container_summary(container).await? {
                container_infos.push(info);
            }
        }
        
        // Sort by status (running first) then by name
        container_infos.sort_by(|a, b| {
            match (a.status.is_running(), b.status.is_running()) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.display_name().cmp(b.display_name()),
            }
        });
        
        Ok(container_infos)
    }
    
    async fn convert_container_summary(&self, summary: ContainerSummary) -> Result<Option<ContainerInfo>> {
        let id = summary.id.unwrap_or_default();
        let names = summary.names.unwrap_or_default();
        let name = names.first().cloned().unwrap_or_else(|| id.clone());
        
        let status = self.parse_container_status(&summary.status.unwrap_or_default(), &summary.state.unwrap_or_default());
        
        let ports = summary.ports.unwrap_or_default()
            .iter()
            .filter_map(|port| {
                if let Some(public_port) = port.public_port {
                    Some(format!("{}:{}", public_port, port.private_port))
                } else {
                    Some(format!("{}", port.private_port))
                }
            })
            .collect();
        
        let created = DateTime::from_timestamp(summary.created.unwrap_or(0), 0)
            .unwrap_or_else(|| Utc::now());
        
        let image = summary.image.unwrap_or_default();
        
        Ok(Some(ContainerInfo {
            id,
            name,
            status,
            ports,
            created,
            started_at: None, // We'll need to get this from inspect if needed
            finished_at: None, // We'll need to get this from inspect if needed
            image,
        }))
    }
    
    fn parse_container_status(&self, status: &str, state: &str) -> ContainerStatus {
        match state.to_lowercase().as_str() {
            "running" => ContainerStatus::Running,
            "exited" => {
                // Try to extract exit code from status
                if let Some(code_str) = status.strip_prefix("Exited (").and_then(|s| s.split(')').next()) {
                    if let Ok(code) = code_str.parse::<i32>() {
                        ContainerStatus::Exited(code)
                    } else {
                        ContainerStatus::Exited(0)
                    }
                } else {
                    ContainerStatus::Exited(0)
                }
            }
            "created" => ContainerStatus::Created,
            "restarting" => ContainerStatus::Restarting,
            "removing" => ContainerStatus::Removing,
            "paused" => ContainerStatus::Paused,
            "dead" => ContainerStatus::Dead,
            _ => ContainerStatus::Exited(0),
        }
    }
    
}