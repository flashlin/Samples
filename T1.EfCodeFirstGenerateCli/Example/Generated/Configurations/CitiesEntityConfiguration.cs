using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CitiesEntityConfiguration : IEntityTypeConfiguration<CitiesEntity>
    {
        public void Configure(EntityTypeBuilder<CitiesEntity> builder)
        {
            builder.ToTable("Cities");

            builder.HasKey(x => x.CityID);

            builder.Property(x => x.CityID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CityName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Status)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedTime)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedBy)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedTime)
                .HasColumnType("timestamp")
                .IsRequired()
            ;

            builder.Property(x => x.CountryID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

        }
    }
}
