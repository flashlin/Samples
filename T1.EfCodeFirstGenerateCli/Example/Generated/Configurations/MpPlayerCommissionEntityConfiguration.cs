using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MpPlayerCommissionEntityConfiguration : IEntityTypeConfiguration<MpPlayerCommissionEntity>
    {
        public void Configure(EntityTypeBuilder<MpPlayerCommissionEntity> builder)
        {
            builder.ToTable("MpPlayerCommission");

            builder.HasKey(x => new { x.CustomerId, x.Type });

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ParentId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Type)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.Commission)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
