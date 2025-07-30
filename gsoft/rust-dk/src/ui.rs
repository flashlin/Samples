use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph, Row, Table},
    Frame, Terminal,
};
use std::io;

use crate::types::ContainerInfo;

pub struct App {
    pub containers: Vec<ContainerInfo>,
    pub filtered_containers: Vec<usize>, // indices into containers
    pub selected: usize,
    pub filter: String,
    pub show_filter: bool,
    pub selected_items: Vec<bool>, // for multi-select
    pub multi_select_mode: bool,
}

impl App {
    pub fn new(containers: Vec<ContainerInfo>, multi_select: bool) -> Self {
        let filtered_containers: Vec<usize> = (0..containers.len()).collect();
        let selected_items = vec![false; containers.len()];
        
        Self {
            containers,
            filtered_containers,
            selected: 0,
            filter: String::new(),
            show_filter: false,
            selected_items,
            multi_select_mode: multi_select,
        }
    }
    
    pub fn next(&mut self) {
        if !self.filtered_containers.is_empty() {
            self.selected = (self.selected + 1) % self.filtered_containers.len();
        }
    }
    
    pub fn previous(&mut self) {
        if !self.filtered_containers.is_empty() {
            if self.selected == 0 {
                self.selected = self.filtered_containers.len() - 1;
            } else {
                self.selected -= 1;
            }
        }
    }
    
    pub fn toggle_selected(&mut self) {
        if let Some(&container_idx) = self.filtered_containers.get(self.selected) {
            self.selected_items[container_idx] = !self.selected_items[container_idx];
        }
    }
    
    pub fn get_selected_container(&self) -> Option<&ContainerInfo> {
        self.filtered_containers
            .get(self.selected)
            .and_then(|&idx| self.containers.get(idx))
    }
    
    pub fn get_selected_containers(&self) -> Vec<ContainerInfo> {
        if self.multi_select_mode {
            self.selected_items
                .iter()
                .enumerate()
                .filter_map(|(idx, &selected)| {
                    if selected {
                        self.containers.get(idx).cloned()
                    } else {
                        None
                    }
                })
                .collect()
        } else if let Some(container) = self.get_selected_container() {
            vec![container.clone()]
        } else {
            vec![]
        }
    }
    
    pub fn update_filter(&mut self, filter: String) {
        self.filter = filter;
        self.apply_filter();
    }
    
    fn apply_filter(&mut self) {
        if self.filter.is_empty() {
            self.filtered_containers = (0..self.containers.len()).collect();
        } else {
            self.filtered_containers = self.containers
                .iter()
                .enumerate()
                .filter(|(_, container)| {
                    container.display_name().to_lowercase().contains(&self.filter.to_lowercase()) ||
                    container.short_id().to_lowercase().contains(&self.filter.to_lowercase())
                })
                .map(|(idx, _)| idx)
                .collect();
        }
        
        // Reset selection to first item
        self.selected = 0;
    }
}

pub async fn run_interactive_selection<F, Fut>(
    containers: Vec<ContainerInfo>,
    title: &str,
    multi_select: bool,
    callback: F,
) -> Result<()>
where
    F: FnOnce(Vec<ContainerInfo>) -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    
    let mut app = App::new(containers, multi_select);
    let result = run_app(&mut terminal, &mut app, title);
    
    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    
    match result {
        Ok(true) => {
            let selected = app.get_selected_containers();
            if !selected.is_empty() {
                callback(selected).await?;
            }
        }
        Ok(false) => {
            // User cancelled
        }
        Err(e) => return Err(e),
    }
    
    Ok(())
}

fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    title: &str,
) -> Result<bool> {
    loop {
        terminal.draw(|f| ui(f, app, title))?;
        
        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                match key.code {
                    KeyCode::Esc if app.show_filter => {
                        app.show_filter = false;
                        app.filter.clear();
                        app.update_filter(String::new());
                    }
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(false),
                    KeyCode::Enter => return Ok(true),
                    KeyCode::Up => app.previous(),
                    KeyCode::Down => app.next(),
                    KeyCode::Char(' ') if app.multi_select_mode => app.toggle_selected(),
                    KeyCode::Char('/') => {
                        app.show_filter = true;
                        app.filter.clear();
                        app.update_filter(String::new());
                    }
                    KeyCode::Char(c) if app.show_filter => {
                        app.filter.push(c);
                        app.update_filter(app.filter.clone());
                    }
                    KeyCode::Char(c) if c.is_alphanumeric() || c == '-' || c == '_' => {
                        // 自動啟動過濾模式
                        app.show_filter = true;
                        app.filter.clear();
                        app.filter.push(c);
                        app.update_filter(app.filter.clone());
                    }
                    KeyCode::Backspace if app.show_filter => {
                        app.filter.pop();
                        app.update_filter(app.filter.clone());
                    }
                    _ => {}
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &App, title: &str) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3), // Title
            Constraint::Min(0),    // Table
            Constraint::Length(3), // Help
        ])
        .split(f.size());
    
    // Title
    let title_block = Block::default()
        .borders(Borders::ALL)
        .title(title);
    let title_paragraph = Paragraph::new(format!("Total: {} containers", app.containers.len()))
        .block(title_block)
        .alignment(Alignment::Center);
    f.render_widget(title_paragraph, chunks[0]);
    
    // Container table
    let header = Row::new(vec!["", "ID", "Name", "Status", "Ports", "Time"])
        .style(Style::default().fg(Color::Yellow))
        .height(1);
    
    let rows: Vec<Row> = app.filtered_containers
        .iter()
        .enumerate()
        .map(|(list_idx, &container_idx)| {
            let container = &app.containers[container_idx];
            let selected_mark = if app.multi_select_mode && app.selected_items[container_idx] {
                "✓"
            } else {
                ""
            };
            
            let style = if list_idx == app.selected {
                Style::default().bg(Color::Blue).fg(Color::White)
            } else if container.status.is_running() {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Red)
            };
            
            let status_string = container.status.display_string();
            let ports_string = container.ports_string();
            let time_string = container.time_string();
            
            Row::new(vec![
                selected_mark.to_string(),
                container.short_id().to_string(),
                container.display_name().to_string(),
                status_string,
                ports_string,
                time_string,
            ])
            .style(style)
        })
        .collect();
    
    let table = Table::new(rows)
        .header(header)
        .block(Block::default().borders(Borders::ALL))
        .widths(&[
            Constraint::Length(2),  // Selection mark
            Constraint::Length(12), // ID
            Constraint::Min(20),    // Name
            Constraint::Length(15), // Status
            Constraint::Length(20), // Ports
            Constraint::Length(15), // Time
        ]);
    
    f.render_widget(table, chunks[1]);
    
    // Help text
    let help_text = if app.show_filter {
        format!("Filter: {} | ESC: clear filter | Enter: select | q: quit", app.filter)
    } else if app.multi_select_mode {
        "↑↓: navigate | Space: toggle | /: filter | Enter: confirm | q: quit".to_string()
    } else {
        "↑↓: navigate | /: filter | Enter: select | q: quit".to_string()
    };
    
    let help = Paragraph::new(help_text)
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
    f.render_widget(help, chunks[2]);
}

pub fn display_container_table(containers: &[ContainerInfo]) -> Result<()> {
    println!("\n{:<12} {:<30} {:<15} {:<20} {:<15}", "ID", "NAME", "STATUS", "PORTS", "TIME");
    println!("{}", "-".repeat(92));
    
    for container in containers {
        let status_color = if container.status.is_running() { "\x1b[32m" } else { "\x1b[31m" };
        let reset_color = "\x1b[0m";
        
        println!(
            "{:<12} {:<30} {}{:<15}{} {:<20} {:<15}",
            container.short_id(),
            container.display_name(),
            status_color,
            container.status.display_string(),
            reset_color,
            container.ports_string(),
            container.time_string()
        );
    }
    
    println!();
    Ok(())
}