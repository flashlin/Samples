using ExpectedObjects;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using NSubstitute;
using NUnit.Framework;
using PizzaWeb.Controllers;
using PizzaWeb.Models;
using PizzaWeb.Models.Banner;
using PizzaWeb.Models.Helpers;
using PizzaWeb.Models.Repos;
using T1.AspNetCore;
using T1.SqlLocalData;
using T1.Standard.IO;

namespace TestProject
{
    public class BannerDbTest
    {
        private readonly SqlLocalDb _localDb = new SqlLocalDb(@"D:\Demo");
        private BannerController _bannerController = default!;
        private string _databaseName = "Northwind";
        private PizzaDbContext _db = default!;
        private string _instanceName = "local_db_instance";

        [SetUp]
        public void Setup()
        {
            _localDb.EnsureInstanceCreated(_instanceName);
            _localDb.ForceDropDatabase(_instanceName, _databaseName);
            _localDb.DeleteDatabaseFile(_databaseName);
            _localDb.CreateDatabase(_instanceName, _databaseName);

            RebuildDatabaseSchema();
        }

        private void RebuildDatabaseSchema()
        {
            var factory = new SqlServerDbContextOptionsFactory(Options.Create(new PizzaDbConfig
            {
                ConnectionString = "Server=(localdb)\\local_db_instance;Integrated security=SSPI;database=Northwind;"
            }));
            _db = new PizzaDbContext(factory.Create());
            var sql = typeof(BannerDbTest).Assembly.GetEmbeddedResourceString("PizzaDb.sql");
            _db.Database.ExecuteSqlRaw(sql);
        }

        [Test]
        public void AddBannerTemplate()
        {
            GivenServiceLocator();
            GivenBannerController();

            WhenAddTemplate();

            var bannerTemplate = _db.BannerTemplates.AsNoTracking().First();
            Assert.That(bannerTemplate.TemplateContent, Is.EqualTo("Hello Banner"));

            var expected =
                "{\"image\":{\"name\":\"image\",\"varType\":\"Image(100,200)\"},\"title\":{\"name\":\"title\",\"varType\":\"String\"}}";
            expected.ToExpectedObject().ShouldEqual(bannerTemplate.VariablesJson);
        }


        [Test]
        public void AddBanner()
        {
            GivenServiceLocator();
            GivenBannerController();

            WhenAddBanner("Template1", "Mother's Day", "SaltedChicken");

            var banner = _db.Banners.AsNoTracking().First();
            var expected =
                "{\"image\":{\"varName\":\"image\",\"resxName\":\"SaltedChickenPizzaImage\"},\"title\":{\"varName\":\"title\",\"resxName\":\"SaltedChickenPizzaTitle\"}}";
            expected.ToExpectedObject().ShouldEqual(banner.VariableOptionsJson);
        }

        private void WhenAddTemplate()
        {
            _bannerController.AddBannerTemplate(new AddBannerTemplateReq()
            {
                TemplateName = "Template1",
                TemplateContent = "Hello Banner",
                Variables = new Dictionary<string, TemplateVariable>()
                {
                    {"image", new TemplateVariable {Name = "image", VarType = "Image(100,200)"}},
                    {"title", new TemplateVariable {Name = "title", VarType = "String"}},
                }
            });
        }

        private void WhenAddBanner(string templateName, string bannerName, string taste)
        {
            _bannerController.AddBanner(new AddBannerReq()
            {
                TemplateName = templateName,
                BannerName = bannerName,
                OrderId = 1,
                VariablesOptions = new Dictionary<string, TemplateVariableValue>()
                {
                    {"image", new TemplateVariableValue {VarName = "image", ResxName = $"{taste}PizzaImage"}},
                    {"title", new TemplateVariableValue {VarName = "title", ResxName = $"{taste}PizzaTitle"}},
                }
            });
        }


        [Test]
        public void GetBannerTemplate()
        {
            GivenServiceLocator();
            GivenBannerController();

            WhenAddTemplate();
            var templates = _bannerController.GetAllTemplates();

            new TemplateVariable
                {
                    Name = "image",
                    VarType = "Image(100,200)"
                }
                .ToExpectedObject()
                .ShouldEqual(templates[0].Variables[0]);
        }


        [Test]
        public void GetBanner()
        {
            GivenServiceLocator();
            GivenBannerController();

            WhenAddTemplate();
            WhenAddBanner("Template1", "Mother Day", "SaltedChicken");
            WhenAddBanner("Template1", "Father Day", "Squid");
            WhenAddResx();

            var banners = _bannerController.GetBanners(new GetBannersReq()
            {
                TemplateName = "Template1"
            });

            new BannerVariable
                {
                    VarName = "image",
                    ResxName = "SaltedChickenPizzaImage",
                    ResxList = new List<VariableResx>(new[]
                    {
                        new VariableResx {IsoLangCode = "en-US", Content = "English Salted Chicken Pizza Url"},
                        new VariableResx {IsoLangCode = "zh-TW", Content = "鹹酥雞披薩圖片連結"},
                    })
                }.ToExpectedObject()
                .ShouldEqual(banners[0].Variables[0]);

            new BannerVariable
                {
                    VarName = "title",
                    ResxName = "SaltedChickenPizzaTitle",
                    ResxList = new List<VariableResx>(new[]
                    {
                        new VariableResx {IsoLangCode = "en-US", Content = "Salted Chicken Pizza"},
                    })
                }
                .ToExpectedObject()
                .ShouldEqual(banners[0].Variables[1]);

            new BannerVariable
                {
                    VarName = "image",
                    ResxName = "SquidPizzaImage",
                    ResxList = new List<VariableResx>(new[]
                    {
                        new VariableResx {IsoLangCode = "en-US", Content = "English Squid Pizza Url"},
                    })
                }.ToExpectedObject()
                .ShouldEqual(banners[1].Variables[0]);
        }

        private void WhenAddResx()
        {
            _db.BannerResx.Add(new BannerResxEntity()
            {
                Name = "SaltedChickenPizzaImage",
                VarType = "Image(100,200)",
                IsoLangCode = "en-US",
                Content = "English Salted Chicken Pizza Url",
            });
            _db.BannerResx.Add(new BannerResxEntity()
            {
                Name = "SaltedChickenPizzaImage",
                VarType = "Image(100,200)",
                IsoLangCode = "zh-TW",
                Content = "鹹酥雞披薩圖片連結",
            });
            _db.BannerResx.Add(new BannerResxEntity()
            {
                Name = "SaltedChickenPizzaTitle",
                VarType = "String",
                IsoLangCode = "en-US",
                Content = "Salted Chicken Pizza",
            });

            _db.BannerResx.Add(new BannerResxEntity()
            {
                Name = "SquidPizzaImage",
                VarType = "Image(100,200)",
                IsoLangCode = "en-US",
                Content = "English Squid Pizza Url",
            });
            _db.BannerResx.Add(new BannerResxEntity()
            {
                Name = "SquidPizzaTitle",
                VarType = "String",
                IsoLangCode = "en-US",
                Content = "Squid Pizza",
            });
            _db.SaveChanges();
        }

        private void GivenBannerController()
        {
            var repo = new PizzaRepo(_db, new JsonConverter());
            var viewToStringRenderer = Substitute.For<IViewToStringRendererService>();
            _bannerController = new BannerController(repo, viewToStringRenderer);
        }

        private static void GivenServiceLocator()
        {
            var services = new ServiceCollection();
            services.AddTransient<IJsonConverter, JsonConverter>();
            var sp = services.BuildServiceProvider();
            ServiceLocator.SetLocatorProvider(sp);
        }
    }
}