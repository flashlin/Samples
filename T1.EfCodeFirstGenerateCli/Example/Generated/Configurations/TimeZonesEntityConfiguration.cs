using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class TimeZonesEntityConfiguration : IEntityTypeConfiguration<TimeZonesEntity>
    {
        public void Configure(EntityTypeBuilder<TimeZonesEntity> builder)
        {
            builder.ToTable("TimeZones");

            builder.HasKey(x => x.TimeZoneID);

            builder.Property(x => x.TimeZoneID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.TimeZoneName)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Presentation)
                .HasColumnType("varchar(100)")
                .IsRequired()
                .HasMaxLength(100)
            ;

            builder.Property(x => x.GMTOffSet)
                .HasColumnType("numeric(4,2)")
                .IsRequired()
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

        }
    }
}
