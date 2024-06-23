mod linebuffer;

use linebuffer::LineBuffer;

fn main() {
    let line1 = LineBuffer::new(30);
    println!("Hello, world! {}", line1.area());
}
