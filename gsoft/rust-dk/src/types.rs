use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerInfo {
    pub id: String,
    pub name: String,
    pub status: ContainerStatus,
    pub ports: Vec<String>,
    pub created: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub image: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerStatus {
    Running,
    Exited(i32), // exit code
    Created,
    Restarting,
    Removing,
    Paused,
    Dead,
}

impl ContainerStatus {
    pub fn is_running(&self) -> bool {
        matches!(self, ContainerStatus::Running)
    }
    
    pub fn display_string(&self) -> String {
        match self {
            ContainerStatus::Running => "Running".to_string(),
            ContainerStatus::Exited(code) => format!("Exited ({})", code),
            ContainerStatus::Created => "Created".to_string(),
            ContainerStatus::Restarting => "Restarting".to_string(),
            ContainerStatus::Removing => "Removing".to_string(),
            ContainerStatus::Paused => "Paused".to_string(),
            ContainerStatus::Dead => "Dead".to_string(),
        }
    }
}

impl ContainerInfo {
    pub fn short_id(&self) -> &str {
        &self.id[..12.min(self.id.len())]
    }
    
    pub fn display_name(&self) -> &str {
        if self.name.starts_with('/') {
            &self.name[1..]
        } else {
            &self.name
        }
    }
    
    pub fn ports_string(&self) -> String {
        if self.ports.is_empty() {
            "-".to_string()
        } else {
            self.ports.join(", ")
        }
    }
    
    pub fn time_string(&self) -> String {
        match &self.status {
            ContainerStatus::Running => {
                if let Some(started) = self.started_at {
                    format_duration_since(started)
                } else {
                    "-".to_string()
                }
            }
            _ => {
                if let Some(finished) = self.finished_at {
                    format_duration_since(finished)
                } else {
                    "-".to_string()
                }
            }
        }
    }
}

fn format_duration_since(time: DateTime<Utc>) -> String {
    let now = Utc::now();
    let duration = now.signed_duration_since(time);
    
    if duration.num_days() > 0 {
        format!("{} days ago", duration.num_days())
    } else if duration.num_hours() > 0 {
        format!("{} hours ago", duration.num_hours())
    } else if duration.num_minutes() > 0 {
        format!("{} minutes ago", duration.num_minutes())
    } else {
        "Just now".to_string()
    }
}