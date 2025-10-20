using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CountryPhoneCodeEntityConfiguration : IEntityTypeConfiguration<CountryPhoneCodeEntity>
    {
        public void Configure(EntityTypeBuilder<CountryPhoneCodeEntity> builder)
        {
            builder.ToTable("CountryPhoneCode");


            builder.Property(x => x.No)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CountryName)
                .HasColumnType("nvarchar(200)")
                .IsRequired()
                .HasMaxLength(200)
            ;

            builder.Property(x => x.PhoneCode)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CountryCode)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Status)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.TStamp)
                .HasColumnType("timestamp")
                .IsRequired()
            ;

            builder.Property(x => x.ContinentCode)
                .HasColumnType("char(2)")
                .HasMaxLength(2)
            ;

            builder.Property(x => x.ContinentName)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Continentorder)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.CurrencyDenied)
                .HasColumnType("varchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.IsLayout)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.IsDefault)
                .HasColumnType("bit")
            ;

        }
    }
}
