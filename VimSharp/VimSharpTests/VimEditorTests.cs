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

        /// <summary>
        /// 測試當游標在文本末尾時，按下向右鍵，游標位置應該保持不變
        /// </summary>
        [Test]
        public void WhenCursorAtEndOfText_PressRightArrow_CursorShouldNotMove()
        {
            // Given
            _editor.Context.SetText(0, 0, "Hello, World!");
            _editor.Context.ViewPort = new ConsoleRectangle(10, 1, 40, 10);
            _editor.Context.CursorX = 13; // 設置游標位置在 '!' 上
            
            // 模擬按下向右鍵
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('\0', ConsoleKey.RightArrow, false, false, false));
            
            // When
            _editor.WaitForInput();
            
            // Then
            _editor.Context.CursorX.Should().Be(13); // 游標位置應該保持不變
        }

        /// <summary>
        /// 測試在普通模式下，按下向右鍵，游標應該向右移動
        /// </summary>
        [Test]
        public void WhenInNormalMode_PressRightArrow_CursorShouldMove()
        {
            // Given
            _editor.Context.SetText(0, 0, "Hello, World!");
            _editor.Context.ViewPort = new ConsoleRectangle(10, 1, 40, 10);
            _editor.Context.CursorX = 13; // 設置游標位置在 '!' 上
            _editor.Mode = new VimNormalMode { Instance = _editor };
            
            // 模擬按下向右鍵
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('\0', ConsoleKey.RightArrow, false, false, false));
            
            // When
            _editor.WaitForInput();
            
            // Then
            _editor.Context.CursorX.Should().Be(14); // 游標位置應該向右移動一格
        }

        /// <summary>
        /// 測試在普通模式下，按下 Esc 鍵，應該切換到視覺模式並將游標向後移動一格
        /// </summary>
        [Test]
        public void WhenInNormalMode_PressEsc_ShouldSwitchToVisualModeAndMoveCursorBack()
        {
            // Given
            _editor.Context.SetText(0, 0, "World!");
            _editor.Context.ViewPort = new ConsoleRectangle(10, 1, 40, 10);
            _editor.Context.CursorX = 6; // 設置游標位置在 '!'後面
            _editor.Mode = new VimNormalMode { Instance = _editor };
            
            // 模擬按下 Esc 鍵
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('\0', ConsoleKey.Escape, false, false, false));
            
            // When
            _editor.WaitForInput();
            
            // Then
            _editor.Context.CursorX.Should().Be(5); // 游標應該在 '!' 上面
            _editor.Mode.Should().BeOfType<VimVisualMode>(); // 模式應該切換到 VimVisualMode
        }

        /// <summary>
        /// 測試在視覺模式下，按下 a 鍵然後按下 Esc 鍵，游標應該向後移動一格
        /// </summary>
        [Test]
        public void WhenInVisualMode_PressA_ThenPressEsc_CursorShouldMoveBackOnePosition()
        {
            // Given
            _editor.Context.SetText(0, 0, "Hello, World!");
            _editor.Context.ViewPort = new ConsoleRectangle(10, 1, 40, 10);
            _editor.Context.CursorX = 13; // 設置游標位置在本文最後一個字上, 例如 "Hello, World!" 的 '!' 上
            _editor.Mode = new VimVisualMode { Instance = _editor };
            
            // 模擬按下 'a' 鍵，切換到 NormalMode 並將游標向右移動一格
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('a', ConsoleKey.A, false, false, false));
            _editor.WaitForInput();
            
            // 此時游標應該在 '!' 後面，模式應該是 NormalMode
            _editor.Context.CursorX.Should().Be(14);
            _editor.Mode.Should().BeOfType<VimNormalMode>();
            
            // 模擬按下 Esc 鍵，切換回 VisualMode 並將游標向左移動一格
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('\0', ConsoleKey.Escape, false, false, false));
            _editor.WaitForInput();
            
            // Then
            _editor.Context.CursorX.Should().Be(13); // 游標應該向左移動一格
            _editor.Mode.Should().BeOfType<VimVisualMode>(); // 模式應該切換回 VimVisualMode
        }

        /// <summary>
        /// 測試在視覺模式下，按下 a 鍵然後按下 Esc 鍵，游標應該向後移動一格（第二種情況）
        /// </summary>
        [Test]
        public void WhenInVisualMode_PressA_ThenPressEsc_CursorShouldMoveBackOnePositionCase2()
        {
            // Given
            _editor.Context.SetText(0, 0, "Hello");
            _editor.Context.ViewPort = new ConsoleRectangle(10, 1, 40, 10);
            _editor.Context.CursorX = 4; // 設置游標位置在本文最後一個字上, 例如 "Hello" 的 'o' 上
            _editor.Mode = new VimVisualMode { Instance = _editor };
            
            // 模擬按下 'a' 鍵，切換到 NormalMode 並將游標向右移動一格
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('a', ConsoleKey.A, false, false, false));
            _editor.WaitForInput();
            
            // 此時游標應該在 'o' 後面，模式應該是 NormalMode
            _editor.Context.CursorX.Should().Be(5);
            _editor.Mode.Should().BeOfType<VimNormalMode>();
            
            // 模擬按下 Esc 鍵，切換回 VisualMode 並將游標向左移動一格
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('\0', ConsoleKey.Escape, false, false, false));
            _editor.WaitForInput();
            
            // Then
            _editor.Context.CursorX.Should().Be(4); // 游標應該向左移動一格
            _editor.Mode.Should().BeOfType<VimVisualMode>(); // 模式應該切換回 VimVisualMode
        }

        /// <summary>
        /// 測試按下向下鍵時，游標應該移動到下一行
        /// </summary>
        [Test]
        public void WhenPressDownArrow_CursorShouldMoveToNextLine()
        {
            // Given
            _editor.Context.SetText(0, 0, "Hello, World!");
            _editor.Context.SetText(0, 1, "123");
            _editor.Context.ViewPort = new ConsoleRectangle(10, 1, 40, 10);
            
            // 模擬按下向右鍵 13 次
            for (int i = 0; i < 13; i++)
            {
                _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('\0', ConsoleKey.RightArrow, false, false, false));
                _editor.WaitForInput();
            }

            
            // 模擬按下向下鍵
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('\0', ConsoleKey.DownArrow, false, false, false));
            
            // When
            _editor.WaitForInput();
            
            // Then
            _editor.Context.CursorY.Should().Be(1); // 游標應該在 "Hello, World!" 的下一行
            _editor.Context.CursorX.Should().Be(2); // 游標應該在 '3' 上面
        }

        /// <summary>
        /// 測試在視覺模式下，按下向上鍵時，游標應該移動到上一行
        /// </summary>
        [Test]
        public void WhenInVisualMode_PressUpArrow_CursorShouldMoveToUpperLine()
        {
            // Given
            _editor.Context.SetText(0, 0, "Hello");
            _editor.Context.SetText(0, 1, "ab");
            _editor.SetViewPort(10, 1, 40, 10);
            _editor.Context.CursorY = 2;
            _editor.Context.CursorX = 11; 
            
            // 模擬按下 UpArrow 鍵
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(new ConsoleKeyInfo('\0', ConsoleKey.UpArrow, false, false, false));
            _editor.WaitForInput();
            
            // Then
            _editor.Context.CursorX.Should().Be(5); // 游標應該在 'o' 
            _editor.Mode.Should().BeOfType<VimVisualMode>(); 
        }

        /// <summary>
        /// 測試將游標移動到 'W' 位置，然後按下向下鍵，游標應該移動到下一行的相同位置
        /// </summary>
        [Test]
        public void WhenMoveCursorToW_ThenPressDownArrow_CursorShouldMoveToSamePositionInNextLine()
        {
            // 初始化 VimEditor
            _editor.Context.Texts.Clear();
            _editor.Context.Texts.Add(new ConsoleText());
            _editor.Context.Texts.Add(new ConsoleText());
            _editor.Context.Texts[0].SetText(0, "Hello, World!");
            _editor.Context.Texts[1].SetText(0, "Ex");
            
            // 設置初始游標位置
            _editor.Context.CursorX = 0;
            _editor.Context.CursorY = 0;
            
            // 按下右鍵按鈕7次，移動到 "W" 的位置
            for (int i = 0; i < 7; i++)
            {
                _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('\0', ConsoleKey.RightArrow, false, false, false));
                _editor.WaitForInput();
            }
            
            // 按下向下按鈕1次
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('\0', ConsoleKey.DownArrow, false, false, false));
            _editor.WaitForInput();
            
            // 驗證游標位置
            _editor.Context.CursorX.Should().Be(1);
            _editor.Context.CursorY.Should().Be(1);
        }

        /// <summary>
        /// 測試在普通模式下按下 $ 鍵時，游標應該移動到行尾
        /// </summary>
        [Test]
        public void WhenPressDollarSign_CursorShouldMoveToEndOfLine()
        {
            // 初始化 VimEditor
            _editor.Context.Texts.Clear();
            _editor.Context.Texts.Add(new ConsoleText());
            _editor.Context.Texts[0].SetText(0, "Hello, World!");
            
            // 設置初始游標位置
            _editor.Context.CursorX = 0;
            _editor.Context.CursorY = 0;
            
            // 按下 '$' 按鍵
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('$', ConsoleKey.D4, true, false, false));
            _editor.WaitForInput();
            
            // 驗證游標位置
            _editor.Context.CursorX.Should().Be(12); // 游標應該在 '!' 上
            _editor.Context.CursorY.Should().Be(0);
        }

        /// <summary>
        /// 測試在啟用相對行號時按下 $ 鍵，游標應該移動到行尾，並考慮相對行號區域的寬度
        /// </summary>
        [Test]
        public void WhenRelativeLineNumberEnabled_PressDollarSign_CursorShouldMoveToEndOfLine()
        {
            // 初始化 VimEditor
            _editor.Context.Texts.Clear();
            _editor.Context.Texts.Add(new ConsoleText());
            _editor.Context.Texts[0].SetText(0, "Hello, World!");
            
            // 設置初始游標位置
            _editor.Context.CursorX = 0;
            _editor.Context.CursorY = 0;

            _editor.IsRelativeLineNumber = true;
            
            // 按下 '$' 按鍵
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('$', ConsoleKey.D4, true, false, false));
            _editor.WaitForInput();
            
            // 驗證游標位置
            _editor.Context.CursorX.Should().Be(14); // 游標應該在 '!' 上
            _editor.Context.CursorY.Should().Be(0);
        }

        [Test]
        public void WhenRelativeLineNumberEnabled_PressCaretSign_CursorShouldMoveToStartOfLine()
        {
            // 初始化 VimEditor
            _editor.Context.Texts.Clear();
            _editor.Context.Texts.Add(new ConsoleText());
            _editor.Context.Texts[0].SetText(0, "Hello, World!");
            
            // 設置初始游標位置
            _editor.Context.CursorX = 10;
            _editor.Context.CursorY = 0;

            _editor.IsRelativeLineNumber = true;
            
            // 按下 '^' 按鍵
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('^', ConsoleKey.D6, true, false, false));
            _editor.WaitForInput();
            
            // 驗證游標位置
            _editor.Context.CursorX.Should().Be(2); // 游標應該在 'H' 上
            _editor.Context.CursorY.Should().Be(0);
        }
        
        [Test]
        public void WhenPressCaretSign_CursorShouldMoveToStartOfLine()
        {
            // 初始化 VimEditor
            _editor.Context.Texts.Clear();
            _editor.Context.Texts.Add(new ConsoleText());
            _editor.Context.Texts[0].SetText(0, "Hello, World!");
            
            // 設置初始游標位置在行尾
            _editor.Context.CursorX = 12;
            _editor.Context.CursorY = 0;
            
            // 按下 '^' 按鍵
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('^', ConsoleKey.D6, true, false, false));
            _editor.WaitForInput();
            
            // 驗證游標位置
            _editor.Context.CursorX.Should().Be(0); // 游標應該在 'H' 上
            _editor.Context.CursorY.Should().Be(0);
        }
        
        /// <summary>
        /// 測試在按下 10J 時，游標應該跳轉到第 10 行
        /// </summary>
        [Test]
        public void WhenPress10J_CursorShouldJumpToLine10()
        {
            // 初始化 VimEditor
            _editor.Context.Texts.Clear();
            
            // 設置11行文本
            for (int i = 0; i < 11; i++)
            {
                _editor.Context.Texts.Add(new ConsoleText());
                _editor.Context.Texts[i].SetText(0, $"line{i+1}");
            }
            
            // 設置視口
            _editor.Context.ViewPort = new ConsoleRectangle(0, 0, 40, 5);
            
            // 設置初始游標位置
            _editor.Context.CursorX = 0;
            _editor.Context.CursorY = 0;
            
            // 依序按下 '1', '0', 'J' 按鍵
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('1', ConsoleKey.D1, false, false, false));
            _editor.WaitForInput();
            
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('0', ConsoleKey.D0, false, false, false));
            _editor.WaitForInput();
            
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('J', ConsoleKey.J, false, false, false));
            _editor.WaitForInput();
            
            // 驗證游標位置
            _editor.Context.CursorX.Should().Be(0); // 游標應該在行首
            _editor.Context.CursorY.Should().Be(4); // 游標應該在視窗的第5行
        }
        
        [Test]
        public void WhenPress1J_CursorShouldJumpToLine1()
        {
            // 初始化 VimEditor
            _editor.Context.Texts.Clear();
            
            // 設置5行文本
            for (int i = 0; i < 5; i++)
            {
                _editor.Context.Texts.Add(new ConsoleText());
                _editor.Context.Texts[i].SetText(0, $"line{i+1}");
            }
            
            // 設置視口
            _editor.Context.ViewPort = new ConsoleRectangle(0, 0, 40, 5);
                        
            // 依序按下 '1', 'J' 按鍵
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('1', ConsoleKey.D1, false, false, false));
            _editor.WaitForInput();
            
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('J', ConsoleKey.J, false, false, false));
            _editor.WaitForInput();
            
            // 驗證游標位置
            _editor.Context.CursorX.Should().Be(0); // 游標應該在行首
            _editor.Context.CursorY.Should().Be(1); // 游標應該在第2行
        }
        
        /// <summary>
        /// 測試在啟用相對行號時，調用 Render 方法不應該改變游標位置
        /// </summary>
        [Test]
        public void WhenRelativeLineNumberEnabled_RenderShouldNotChangeCursorPosition()
        {
            // 初始化 VimEditor
            _editor.Context.Texts.Clear();
            
            // 設置5行文本
            for (int i = 0; i < 5; i++)
            {
                _editor.Context.Texts.Add(new ConsoleText());
                _editor.Context.Texts[i].SetText(0, $"line{i+1}");
            }
            
            // 設置視口
            _editor.Context.ViewPort = new ConsoleRectangle(0, 0, 40, 5);
            
            // 設置初始游標位置
            _editor.Context.CursorX = 0;
            _editor.Context.CursorY = 0;
            
            // 啟用相對行號
            _editor.IsRelativeLineNumber = true;
            
            // 調用Render方法
            _editor.Render();
            
            // 驗證游標位置應該在當 IsRelativeLineNumber 為 true 時，游標位置應該在第2行
            _editor.Context.CursorX.Should().Be(2); 
            _editor.Context.CursorY.Should().Be(0); 
        }
        
        [Test]
        public void WhenRelativeLineNumberEnabled_PressLeftArrowTwice_CursorShouldStayAtPosition2()
        {
            // 初始化 VimEditor
            _editor.Context.Texts.Clear();
            
            // 設置5行文本
            for (int i = 0; i < 5; i++)
            {
                _editor.Context.Texts.Add(new ConsoleText());
                _editor.Context.Texts[i].SetText(0, $"line{i+1}");
            }
            
            // 設置視口
            _editor.Context.ViewPort = new ConsoleRectangle(0, 0, 40, 5);
            
            // 設置初始游標位置
            _editor.Context.CursorX = 0;
            _editor.Context.CursorY = 0;
            
            // 啟用相對行號
            _editor.IsRelativeLineNumber = true;
            
            // 按下向左鍵兩次
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('\0', ConsoleKey.LeftArrow, false, false, false));
            _editor.WaitForInput();
            
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('\0', ConsoleKey.LeftArrow, false, false, false));
            _editor.WaitForInput();
            
            // 驗證游標位置
            _editor.Context.CursorX.Should().Be(2); // 游標X位置應該保持在2
            _editor.Context.CursorY.Should().Be(0); // 游標Y位置應該保持不變
        }

        [Test]
        public void WhenStatusBarVisible_PressDownArrow_CursorShouldStopAtLastVisibleLine()
        {
            // 初始化 _editor Texts 10 行內容
            _editor.Context.Texts.Clear();
            for (int i = 0; i < 10; i++)
            {
                _editor.Context.Texts.Add(new ConsoleText());
                _editor.Context.Texts[i].SetText(0, $"Line {i + 1}");
            }
            
            // 設置 ViewPort 
            _editor.SetViewPort(0, 1, 40, 5);
            _editor.IsStatusBarVisible = true;
            
            // 設定完 ViewPort 後，游標應該在第1行
            _editor.Context.CursorX.Should().Be(0);
            _editor.Context.CursorY.Should().Be(1);
            
            // 確保使用 VimVisualMode
            _editor.Mode = new VimVisualMode { Instance = _editor };
            
            // 按下向下按鍵 5 次
            for (int i = 0; i < 15; i++)
            {
                _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('\0', ConsoleKey.DownArrow, false, false, false));
                _editor.WaitForInput();
            }

            _editor.Render();

            // 最終驗證：游標應該停在最後一個可見行（索引為 4）
            _editor.Context.CursorY.Should().Be(4);
        }
    }
} 