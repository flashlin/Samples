using ExpectedObjects;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using NUnit.Framework;
using PizzaWeb.Controllers;
using PizzaWeb.Models;
using PizzaWeb.Models.Banner;
using PizzaWeb.Models.Helpers;
using PizzaWeb.Models.Libs;
using PizzaWeb.Models.Repos;
using T1.SqlLocalData;
using T1.Standard.IO;

namespace TestProject
{
    public class BannerDbTest
    {
        private readonly SqlLocalDb _localDb = new SqlLocalDb(@"D:\Demo");
        private BannerController _bannerController;
        private string _databaseName = "Northwind";
        private PizzaDbContext _db;
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
            var sql = EmbeddedResource.GetEmbeddedResourceString(typeof(BannerDbTest).Assembly, "PizzaDb.sql");
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

            WhenAddBanner("Mother's Day");

            var banner = _db.Banners.AsNoTracking().First();
            var expected =
                "{\"image\":{\"varName\":\"image\",\"resxName\":\"Salted Chicken Pizza\"},\"title\":{\"varName\":\"title\",\"resxName\":\"Mother\\u0027s Chicken\"}}";
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

        private void WhenAddBanner(string bannerName)
        {
            _bannerController.AddBanner(new AddBannerReq()
            {
                TemplateName = "Banner1",
                BannerName = bannerName,
                OrderId = 1,
                VariablesOptions = new Dictionary<string, TemplateVariableValue>()
                {
                    {"image", new TemplateVariableValue {VarName = "image", ResxName = "Salted Chicken Pizza"}},
                    {"title", new TemplateVariableValue {VarName = "title", ResxName = $"{bannerName} Chicken"}},
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

            templates[0].Variables[0].ToExpectedObject()
                .ShouldEqual(new TemplateVariable
                {
                    Name = "image",
                    VarType = "Image(100,200)"
                });
        }
        
        
        [Test]
        public void GetBanner()
        {
            GivenServiceLocator();
            GivenBannerController();

            WhenAddTemplate();
            WhenAddBanner("Mother Day");
            WhenAddBanner("Father Day");
            
            var banners = _bannerController.GetBanners(new GetBannersReq()
            {
                TemplateName = "Template1"
            });

            banners[0].Variables[0].ToExpectedObject()
                .ShouldEqual(new BannerVariable
                {
                    VarName = "image",
                    ResxName = ""
                });
        }

        private void GivenBannerController()
        {
            _bannerController = new BannerController(_db, new JsonConverter(), null);
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