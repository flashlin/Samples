﻿using Microsoft.AspNetCore.Mvc;
using QueryApp.Models;
using QueryKits.Services;

namespace QueryApp.Controllers;

public class HomeController : Controller
{
    private readonly ILocalEnvironment _localEnvironment;

    public HomeController(ILocalEnvironment localEnvironment)
    {
        _localEnvironment = localEnvironment;
    }
    
    public IActionResult Index()
    {
        //return Ok($"Hello {_localEnvironment.AppUid} {_localEnvironment.AppLocation} {_localEnvironment.Port}");
        return View("Index");
    }

    public IActionResult MyBlazor()
    {
        return View("MyBlazor");
    }
}