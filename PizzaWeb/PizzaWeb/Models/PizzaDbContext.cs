using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.ChangeTracking;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;
using Microsoft.Extensions.Options;
using PizzaWeb.Models.Banner;
using PizzaWeb.Models.Helpers;
using T1.Standard.Serialization;

namespace PizzaWeb.Models
{
	public class PizzaDbContext : DbContext
	{
		private readonly IJsonConverter _json;

		public PizzaDbContext(DbContextOptions options)
			 : base(options)
		{
			this._json = new JsonConverter();
		}

		public DbSet<BannerTemplateEntity> BannerTemplates => Set<BannerTemplateEntity>();

		public DbSet<StoreShelvesEntity> StoreShelves => Set<StoreShelvesEntity>();
		public DbSet<BannerEntity> Banners => Set<BannerEntity>();
		public DbSet<BannerResxEntity> BannerResx => Set<BannerResxEntity>();
		public DbSet<BannerShelfEntity> BannerShelf => Set<BannerShelfEntity>();
		public DbSet<VariableShelfEntity> VariableShelf => Set<VariableShelfEntity>();

		protected override void OnModelCreating(ModelBuilder modelBuilder)
		{
			var templateVariablesConverter = new ValueConverter<List<TemplateVariable>, String>(
				  model =>_json.Serialize(model.ToDictionary(x => x.VarName)),
				  value => _json.Deserialize<Dictionary<string, TemplateVariable>>(value).Values.ToList());

			var templateVariablesComparer = new ValueComparer<List<TemplateVariable>>(
				 (l, r) => _json.Serialize(l) == _json.Serialize(r),
				 v => v == null ? 0 : _json.Serialize(v).GetHashCode(),
				 v => _json.Deserialize<List<TemplateVariable>>(_json.Serialize(v)));


			modelBuilder.Entity<BannerTemplateEntity>(builder =>
			{
				builder.Property(x => x.Variables)
					.HasConversion(templateVariablesConverter, templateVariablesComparer);
			});


			//modelBuilder.Entity<BannerTemplateEntity>(builder =>
			//{
			//	builder.Property(x => x.Variables)
			//			 .HasConversion<TemplateVariableConverter, JsonComparer<List<TemplateVariable>>>();
			//});
			modelBuilder.Entity<BannerEntity>(builder =>
			{
				builder.Property(x => x.VariableOptions)
						 .HasConversion<VariableOptionListConverter, JsonComparer<List<VariableOption>>>();
			});
		}
	}

	public static class ValueConversionExtensions
	{
		public static PropertyBuilder<T> HasJsonConversion<T>(this PropertyBuilder<T> propertyBuilder,
			 IJsonConverter json)
		{
			var converter = new ValueConverter<T, String>(
				 v => json.Serialize(v),
				 v => json.Deserialize<T>(v));

			var comparer = new ValueComparer<T>(
				 (l, r) => json.Serialize(l) == json.Serialize(r),
				 v => v == null ? 0 : json.Serialize(v).GetHashCode(),
				 v => json.Deserialize<T>(json.Serialize(v)));

			propertyBuilder.HasConversion(converter);
			propertyBuilder.Metadata.SetValueConverter(converter);
			propertyBuilder.Metadata.SetValueComparer(comparer);
			return propertyBuilder;
		}
	}
}