---
skill: assert-csharp
description: 對於測試規範使用 NUnit 測試框架，應使用 FluentAssertions 函式庫進行所有測試結果的斷言。當驗收結果時，要用 FluentAssertions 的 Should().BeEquivalentTo 方式直接比對預期結果
tags: [csharp, testing, nunit, fluent-assertions]
---

# C# NUnit + FluentAssertions 測試斷言規範

## 核心原則

1. **測試框架：** 使用 NUnit
2. **斷言庫：** 使用 FluentAssertions 6.12.0
3. **斷言方式：** 使用 `Should().BeEquivalentTo()` 進行結果驗證
4. **測試模式：** 嚴格遵守 Arrange-Act-Assert (AAA) 模式

## Context

### 1. 測試框架 (NUnit) 規範

* **命名空間和屬性：** 測試類別必須使用 `[TestFixture]` 或 `[TestFixtureSource]` 屬性；測試方法必須使用 `[Test]` 屬性。
* **Arrange-Act-Assert 模式：** 嚴格遵守 AAA 模式，保持測試方法的結構清晰。

### 2. BeEquivalentTo 的強制使用與選項

對於複雜物件和集合的驗證，**必須**使用 `Should().BeEquivalentTo()`。

**範例：**

```csharp
using NUnit.Framework;
using FluentAssertions;
using System.Collections.Generic;

[TestFixture]
public class MyServiceTests
{
    [Test]
    public void GetDataModels_ShouldReturnCorrectDataStructure()
    {
        // Arrange
        var sut = new MyService();

        // Act
        var actualResult = sut.GetDataModels();

        // Assert (使用 BeEquivalentTo 進行結構性比對)
        actualResult.Should().BeEquivalentTo( /* 預期結果 */,
            options => options
                .ExcludingMissingMembers()
                .Excluding(x => x.CreationDate)
        );
    }
}
```

## 常用配置選項

### 忽略屬性

```csharp
// 忽略特定屬性
result.Should().BeEquivalentTo(expected,
    options => options.Excluding(x => x.Id));

// 忽略多個屬性
result.Should().BeEquivalentTo(expected,
    options => options
        .Excluding(x => x.Id)
        .Excluding(x => x.CreatedDate));
```

### 忽略缺失的成員

```csharp
// 允許實際結果有額外屬性
result.Should().BeEquivalentTo(expected,
    options => options.ExcludingMissingMembers());
```

### 集合順序

```csharp
// 忽略集合順序
result.Should().BeEquivalentTo(expected,
    options => options.WithoutStrictOrdering());
```

### 型別比對

```csharp
// 使用執行時期型別比對
result.Should().BeEquivalentTo(expected,
    options => options.RespectingRuntimeTypes());
```

## 最佳實踐

1. **優先使用 BeEquivalentTo：** 對於物件和集合的比對，始終使用 `BeEquivalentTo()` 而非手動逐一比對屬性
2. **清晰的測試結構：** 使用註解標記 Arrange、Act、Assert 三個區段
3. **有意義的測試名稱：** 測試方法名稱應清楚描述測試情境和預期結果
4. **配置選項文件化：** 當使用特殊配置選項時，加入註解說明原因

## 範例模板

```csharp
using NUnit.Framework;
using FluentAssertions;

[TestFixture]
public class ServiceTests
{
    [Test]
    public void Method_WhenCondition_ShouldExpectedBehavior()
    {
        // Arrange
        var dependency = new Dependency();
        var sut = new Service(dependency);
        var input = new Input();

        // Act
        var actual = sut.Method(input);

        // Assert
        var expected = new ExpectedResult();
        actual.Should().BeEquivalentTo(expected);
    }
}
```
