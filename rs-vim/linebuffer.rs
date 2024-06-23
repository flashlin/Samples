pub struct LineBuffer {
    pub width: u32,
}

impl LineBuffer {
    pub fn new(width: u32) -> LineBuffer {
        LineBuffer { width }
    }

    pub fn area(&self) -> u32 {
        self.width
    }
}

