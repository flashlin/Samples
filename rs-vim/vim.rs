struct LineBuffer {
    width: u32,
}

impl LineBuffer {
    fn new(width: u32) -> LineBuffer {
        LineBuffer { width }
    }

    fn area(&self) -> u32 {
        self.width
    }
}

fn main() {
    let line1 = LineBuffer::new(30);
    println!("Hello, world! {}", line1.area());
}
