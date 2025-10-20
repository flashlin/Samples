using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SettlementLockEntityConfiguration : IEntityTypeConfiguration<SettlementLockEntity>
    {
        public void Configure(EntityTypeBuilder<SettlementLockEntity> builder)
        {
            builder.ToTable("SettlementLock");


            builder.Property(x => x.IsLock)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.OperatorName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Action)
                .HasColumnType("nvarchar(200)")
                .IsRequired()
                .HasMaxLength(200)
            ;

            builder.Property(x => x.Description)
                .HasColumnType("nvarchar(500)")
                .IsRequired()
                .HasMaxLength(500)
            ;

            builder.Property(x => x.RequestDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
