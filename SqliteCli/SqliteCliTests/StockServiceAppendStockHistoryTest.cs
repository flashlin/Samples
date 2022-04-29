using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using FluentAssertions;
using NSubstitute;
using SqliteCli.Entities;
using SqliteCli.Repos;
using Xunit;

namespace SqliteCliTests;

public class StockServiceAppendStockHistoryTest
{
    private IStockRepo _stockRepo;
    private StockService _service;
    private string _anyStockId;

    public StockServiceAppendStockHistoryTest()
    {
        _stockRepo = Substitute.For<IStockRepo>();
        _service = new StockService(_stockRepo, Substitute.For<IStockExchangeApi>());
        _anyStockId = "anyStockId";
    }

    [Fact]
    public async Task not_contain_today()
    {
        var appendStockHistoryResult = WhenReceived<StockHistoryEntity>(_stockRepo, 
            x => x.AppendStockHistory(null!));

        await _service.AppendStockHistoryRangeFromApi(
            DateTime.Parse("2022-04-03"),
            DateTime.Parse("2022-04-05"),
            _anyStockId);


        appendStockHistoryResult.Should().BeEquivalentTo(new[]
        {
            ExpectedStockHistoryEntity("2022-04-04"),
            ExpectedStockHistoryEntity("2022-04-05"),
        });
    }
	
    [Fact]
    public async Task contain_today()
    {
        var appendStockHistoryResult = WhenReceived<StockHistoryEntity>(_stockRepo, 
            x => x.AppendStockHistory(null!));

        await _service.AppendStockHistoryRangeFromApi(
            DateTime.Now,
            DateTime.Now,
            _anyStockId);

        appendStockHistoryResult.Should().BeEmpty();
    }

    private StockHistoryEntity ExpectedStockHistoryEntity(string date)
    {
        return new StockHistoryEntity
        {
            TranDate = DateTime.Parse(date).ToDate(),
            StockId = _anyStockId,
        };
    }

    private static List<T> WhenReceived<T>(IStockRepo stockRepo, Action<IStockRepo> methodPredecate)
    {
        var usedFor = new List<T>();
        stockRepo.WhenForAnyArgs(methodPredecate)
            .Do(x => usedFor.Add(x.Arg<T>()));
        return usedFor;
    }
}