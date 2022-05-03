using Microsoft.AspNetCore.Mvc;
using SqliteCli.Repos;

namespace SqliteCli.Controllers;

public class HomeController : Controller
{
    public IActionResult Index()
    {
        return Content("Hello World");
    }
}

[Route("[controller]/[action]")]
[ApiController]
public class StockController : ControllerBase
{
    private IStockService _stockService;

    public StockController(IStockService stockService)
    {
        _stockService = stockService;
    }
    
    [HttpGet]
    public List<TransHistory> GetTransList()
    {
        return _stockService.GetTransList(new ListTransReq());    
    }
}