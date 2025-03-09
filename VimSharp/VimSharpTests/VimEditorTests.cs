using NUnit.Framework;
using FluentAssertions;
using NSubstitute;
using VimSharpLib;

namespace VimSharpTests
{
    [TestFixture]
    public class VimEditorTests
    {
        private VimEditor _editor;
        private IConsoleDevice _mockConsole;

        [SetUp]
        public void Setup()
        {
            // 創建 Mock 的 IConsoleDevice
            _mockConsole = Substitute.For<IConsoleDevice>();
            _mockConsole.WindowWidth.Returns(80);
            _mockConsole.WindowHeight.Returns(25);
            
            // 創建使用 Mock 控制台的 VimEditor
            _editor = new VimEditor(_mockConsole);
        }

        [Test]
        public void Constructor_ShouldInitializeProperties()
        {
            // Assert
            _editor.IsRunning.Should().BeTrue();
            _editor.Context.Should().NotBeNull();
            _editor.Mode.Should().NotBeNull();
            _editor.Mode.Should().BeOfType<VimVisualMode>();
            _editor.IsStatusBarVisible.Should().BeFalse();
            _editor.StatusBarText.Should().BeEmpty();
        }

        [Test]
        public void Initialize_ShouldSetupContext()
        {
            // Act
            _editor.Initialize();

            // Assert
            _editor.Context.Texts.Should().NotBeEmpty();
            _editor.Context.ViewPort.Should().NotBeNull();
            _editor.Context.ViewPort.Width.Should().Be(80);
            _editor.Context.ViewPort.Height.Should().Be(25);
            
            // 驗證是否調用了 WindowWidth 和 WindowHeight
            _ = _mockConsole.Received(1).WindowWidth;
            _ = _mockConsole.Received(1).WindowHeight;
        }

        [Test]
        public void SetHorizontalOffset_ShouldUpdateOffsetX()
        {
            // Arrange
            int expectedOffset = 5;

            // Act
            _editor.SetHorizontalOffset(expectedOffset);

            // Assert
            _editor.Context.OffsetX.Should().Be(expectedOffset);
        }

        [Test]
        public void SetVerticalOffset_ShouldUpdateOffsetY()
        {
            // Arrange
            int expectedOffset = 3;

            // Act
            _editor.SetVerticalOffset(expectedOffset);

            // Assert
            _editor.Context.OffsetY.Should().Be(expectedOffset);
        }

        [Test]
        public void Scroll_ShouldUpdateOffsets()
        {
            // Arrange
            _editor.Context.OffsetX = 2;
            _editor.Context.OffsetY = 2;
            int deltaX = 3;
            int deltaY = 4;

            // Act
            _editor.Scroll(deltaX, deltaY);

            // Assert
            _editor.Context.OffsetX.Should().Be(5); // 2 + 3
            _editor.Context.OffsetY.Should().Be(6); // 2 + 4
        }

        [Test]
        public void Mode_ShouldBeInitializedAsVimVisualMode()
        {
            // Assert
            _editor.Mode.Should().BeOfType<VimVisualMode>();
            var visualMode = _editor.Mode as VimVisualMode;
            visualMode.Should().NotBeNull();
            visualMode!.Instance.Should().Be(_editor);
        }
        
        [Test]
        public void Render_ShouldUseConsoleDevice()
        {
            // Arrange
            _editor.Context.Texts.Add(new ConsoleText());
            
            // Act
            _editor.Render();
            
            // Assert
            // 驗證是否調用了 SetCursorPosition 和 Write 方法
            _mockConsole.Received().SetCursorPosition(Arg.Any<int>(), Arg.Any<int>());
            _mockConsole.Received().Write(Arg.Any<string>());
        }
        
        [Test]
        public void WaitForInput_ShouldUseConsoleDevice()
        {
            // Arrange
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('a', ConsoleKey.A, false, false, false));
            
            // Act
            _editor.WaitForInput();
            
            // Assert
            // 驗證是否調用了 ReadKey 方法
            _mockConsole.Received().ReadKey(Arg.Any<bool>());
        }

        [Test]
        public void WhenCursorAtEndOfText_PressRightArrow_CursorShouldNotMove()
        {
            // Given
            _editor.Context.SetText(0, 0, "Hello, World!");
            _editor.Context.ViewPort = new ConsoleRectangle(10, 1, 40, 10);
            _editor.Context.CursorX = 14; // 設置游標位置
            
            // 模擬按下向右鍵
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('\0', ConsoleKey.RightArrow, false, false, false));
            
            // When
            _editor.WaitForInput();
            
            // Then
            _editor.Context.CursorX.Should().Be(14); // 游標位置應該保持不變
        }
    }
} 