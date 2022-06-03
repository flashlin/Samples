using ExpectedObjects;
using FluentAssertions;
using FluentAssertions.Equivalency;
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
using ServiceStack;
using T1.AspNetCore;
using T1.SqlLocalData;
using T1.Standard.IO;
using BannerVariable = PizzaWeb.Models.Banner.BannerVariable;

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
                //ConnectionString = "Server=localhost\\SQLEXPRESS;Integrated security=SSPI;database=PizzaDb;"
            }));
            _db = new PizzaDbContext(factory.Create());
            ExecuteEmbeddedSql("PizzaDb.sql");
            ExecuteEmbeddedSql("SP_GetResxNames.sql");

            //             sql = @"
// delete [dbo].[BannerTemplate]
// delete [dbo].[Banner]
// delete [dbo].[Resx]
// delete [dbo].[BannerShelf]
// delete [dbo].[VariableShelf]
// ";
//             _db.Database.ExecuteSqlRaw(sql);
        }

        private void ExecuteEmbeddedSql(string resourceSqlName)
        {
            var sql = typeof(BannerDbTest).Assembly.GetEmbeddedResourceString(resourceSqlName);
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

            var expected = new[]
            {
                new TemplateVariable()
                {
                    VarName = "image",
                    VarType = "Image(100,200)"
                },
                new TemplateVariable()
                {
                    VarName = "title",
                    VarType = "String"
                },
            };

            expected.Should().BeEquivalentTo(bannerTemplate.Variables);
        }


        [Test]
        public void AddBanner()
        {
            GivenServiceLocator();
            GivenBannerController();

            WhenAddBanner("Template1", "Mother's Day", "SaltedChicken");

            var banner = _db.Banners.AsNoTracking().First();

            var expected = new[]
            {
                new VariableOption
                {
                    VarName = "image",
                    ResxName = "SaltedChickenPizzaImage"
                },
                new VariableOption
                {
                    VarName = "title",
                    ResxName = "SaltedChickenPizzaTitle"
                },
            };
            expected.Should().BeEquivalentTo(banner.VariableOptions);
        }

        [Test]
        public void GetBannerTemplate()
        {
            GivenServiceLocator();
            GivenBannerController();

            WhenAddTemplate();
            var templates = _bannerController.GetAllTemplates(new GetBannerTemplatesReq()
            {
                PageSize = 10,
            });

            new TemplateVariable
                {
                    VarName = "image",
                    VarType = "Image(100,200)"
                }
                .ToExpectedObject()
                .ShouldEqual(templates[0].Variables[0]);
        }

        [Test]
        public void GetBannerSettings()
        {
            GivenServiceLocator();
            GivenBannerController();

            WhenAddTemplate();
            WhenAddBanner("Template1", "Mother Day", "SaltedChicken");
            WhenAddBanner("Template1", "Father Day", "Squid");
            WhenAddResx();

            var banners = _bannerController.GetBannerSettings(new GetBannersSettingReq()
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

        [Test]
        public void ApplyBanners()
        {
            GivenServiceLocator();
            GivenBannerController();

            WhenAddTemplate();
            WhenAddBanner("Template1", "Mother Day", "SaltedChicken");
            WhenAddBanner("Template1", "Father Day", "Squid");
            WhenAddResx();

            _bannerController.ApplyBanner(new ApplyBannerReq()
            {
                BannerName = "Mother Day",
            });

            var bannerShelf =
                (from tb1 in _db.BannerShelf.AsNoTracking() select tb1)
                .ToList();

            var variableShelf =
                (from tb1 in _db.VariableShelf.AsNoTracking() select tb1)
                .ToList();

            new[]
                {
                    new VariableShelfEntity()
                    {
                        VarName = "image",
                        Content = "English Salted Chicken Pizza Url",
                        ResxName = "SaltedChickenPizzaImage",
                        IsoLangCode = "en-US",
                    },
                    new VariableShelfEntity()
                    {
                        VarName = "image",
                        Content = "鹹酥雞披薩圖片連結",
                        ResxName = "SaltedChickenPizzaImage",
                        IsoLangCode = "zh-TW",
                    },
                    new VariableShelfEntity()
                    {
                        VarName = "title",
                        Content = "Salted Chicken Pizza",
                        ResxName = "SaltedChickenPizzaTitle",
                        IsoLangCode = "en-US",
                    }
                }.Should()
                .BeEquivalentTo(variableShelf, ExcludeProperties);
        }

        [Test]
        public void GetBannersData()
        {
            GivenServiceLocator();
            GivenBannerController();

            WhenAddTemplate();
            WhenAddBanner("Template1", "Mother Day", "SaltedChicken");
            WhenAddBanner("Template1", "Father Day", "Squid");
            WhenAddResx();

            _bannerController.ApplyBanner(new ApplyBannerReq()
            {
                BannerName = "Mother Day",
            });

            var banners = _bannerController.GetBannersData(new GetBannersDataReq()
            {
                BannerName = "Mother Day",
                IsoLangCode = "en-US",
            });

            new[]
                {
                    new
                    {
                        BannerName = "Mother Day",
                        TemplateName = "Template1",
                        TemplateContent = "Hello Banner",
                        Variables = new[]
                        {
                            new
                            {
                                VarName = "image",
                                ResxName = "SaltedChickenPizzaImage",
                                Content = "English Salted Chicken Pizza Url",
                            },
                            new
                            {
                                VarName = "title",
                                ResxName = "SaltedChickenPizzaTitle",
                                Content = "Salted Chicken Pizza",
                            }
                        }
                    },
                }
                .Should()
                .BeEquivalentTo(banners, ExcludeProperties);
        }

        private void WhenAddBanner(string templateName, string bannerName, string taste)
        {
            _bannerController.AddBanner(new AddBannerReq()
            {
                TemplateName = templateName,
                BannerName = bannerName,
                OrderId = 1,
                VariablesOptions = new Dictionary<string, VariableOption>()
                {
                    {"image", new VariableOption {VarName = "image", ResxName = $"{taste}PizzaImage"}},
                    {"title", new VariableOption {VarName = "title", ResxName = $"{taste}PizzaTitle"}},
                }
            });
        }

        private EquivalencyAssertionOptions<BannerTemplateData> ExcludeProperties(
            EquivalencyAssertionOptions<BannerTemplateData> options)
        {
            options.Excluding(t => t.Uid);
            return options;
        }

        private EquivalencyAssertionOptions<VariableShelfEntity> ExcludeProperties(
            EquivalencyAssertionOptions<VariableShelfEntity> options)
        {
            options.Excluding(t => t.Id);
            options.Excluding(t => t.Uid);
            return options;
        }


        private void WhenAddTemplate()
        {
            _bannerController.AddTemplate(new AddTemplateReq()
            {
                TemplateName = "Template1",
                TemplateContent = "Hello Banner",
                Variables = new Dictionary<string, TemplateVariable>()
                {
                    {"image", new TemplateVariable {VarName = "image", VarType = "Image(100,200)"}},
                    {"title", new TemplateVariable {VarName = "title", VarType = "String"}},
                }
            });
        }

        private void WhenAddResx()
        {
            _db.BannerResx.Add(new BannerResxEntity()
            {
                ResxName = "SaltedChickenPizzaImage",
                VarType = "Image(100,200)",
                IsoLangCode = "en-US",
                Content = "English Salted Chicken Pizza Url",
            });
            _db.BannerResx.Add(new BannerResxEntity()
            {
                ResxName = "SaltedChickenPizzaImage",
                VarType = "Image(100,200)",
                IsoLangCode = "zh-TW",
                Content = "鹹酥雞披薩圖片連結",
            });
            _db.BannerResx.Add(new BannerResxEntity()
            {
                ResxName = "SaltedChickenPizzaTitle",
                VarType = "String",
                IsoLangCode = "en-US",
                Content = "Salted Chicken Pizza",
            });

            _db.BannerResx.Add(new BannerResxEntity()
            {
                ResxName = "SquidPizzaImage",
                VarType = "Image(100,200)",
                IsoLangCode = "en-US",
                Content = "English Squid Pizza Url",
            });
            _db.BannerResx.Add(new BannerResxEntity()
            {
                ResxName = "SquidPizzaTitle",
                VarType = "String",
                IsoLangCode = "en-US",
                Content = "Squid Pizza",
            });
            _db.SaveChanges();
        }

        private static void GivenServiceLocator()
        {
            var services = new ServiceCollection();
            services.AddTransient<IJsonConverter, JsonConverter>();
            var sp = services.BuildServiceProvider();
            ServiceLocator.SetLocatorProvider(sp);
        }

        private void GivenBannerController()
        {
            var repo = new PizzaRepo(_db, new JsonConverter());
            var viewToStringRenderer = Substitute.For<IViewToStringRendererService>();
            _bannerController = new BannerController(repo, viewToStringRenderer);
        }
    }
}