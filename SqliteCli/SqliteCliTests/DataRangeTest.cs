using System;
using System.Linq;
using Microsoft.EntityFrameworkCore.Metadata.Internal;
using SqliteCli;
using SqliteCli.Helpers;
using Xunit;

namespace SqliteCliTests;

public class DataRangeTest
{
    [Fact]
    public void Test()
    {
        var dateRange = new DateRange
        {
            StartDate = DateTime.Parse("2022/01/01"),
            EndDate = DateTime.Parse("2022/04/02")
        };

        var expectedRange = new[]
        {
            DateTime.Parse("2022/01/01"),
            DateTime.Parse("2022/02/01"),
            DateTime.Parse("2022/03/01"),
            DateTime.Parse("2022/04/01"),
        };

        var actualDateRange = dateRange.GetRangeByMonth().ToArray();
        Assert.Equal(expectedRange, actualDateRange);
    }
}