using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerTrackingEntityConfiguration : IEntityTypeConfiguration<CustomerTrackingEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerTrackingEntity> builder)
        {
            builder.ToTable("CustomerTracking");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.TrackingId)
                .HasColumnType("varchar(21)")
                .IsRequired()
                .HasMaxLength(21)
            ;

        }
    }
}
