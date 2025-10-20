using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CountriesEntityConfiguration : IEntityTypeConfiguration<CountriesEntity>
    {
        public void Configure(EntityTypeBuilder<CountriesEntity> builder)
        {
            builder.ToTable("Countries");

            builder.HasKey(x => x.CountryID);

            builder.Property(x => x.CountryID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CountryName)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.NativeName)
                .HasColumnType("nvarchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.ISO3166)
                .HasColumnType("char(2)")
                .IsRequired()
                .HasMaxLength(2)
            ;

            builder.Property(x => x.FIFACode)
                .HasColumnType("char(3)")
                .HasMaxLength(3)
            ;

            builder.Property(x => x.TelephoneCode)
                .HasColumnType("varchar(5)")
                .HasMaxLength(5)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("nvarchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.Status)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(1)
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(1)
            ;

            builder.Property(x => x.CreatedTime)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedBy)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(1)
            ;

            builder.Property(x => x.LastModifiedTime)
                .HasColumnType("timestamp")
                .IsRequired()
            ;

            builder.Property(x => x.Timezone)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ContinentCode)
                .HasColumnType("varchar(2)")
                .HasMaxLength(2)
            ;

            builder.Property(x => x.ContinentName)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ContinentOrder)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.isLeagueOnly)
                .HasColumnType("bit")
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.isDefault)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.isLayout)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.IsEuropeanUnion)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.RiskLevel)
                .HasColumnType("tinyint(3,0)")
            ;

        }
    }
}
