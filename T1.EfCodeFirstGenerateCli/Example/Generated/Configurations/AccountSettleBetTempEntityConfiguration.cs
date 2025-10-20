using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AccountSettleBetTempEntityConfiguration : IEntityTypeConfiguration<AccountSettleBetTempEntity>
    {
        public void Configure(EntityTypeBuilder<AccountSettleBetTempEntity> builder)
        {
            builder.ToTable("AccountSettleBetTemp");

            builder.HasKey(x => new { x.BatchId, x.TransId });

            builder.Property(x => x.BatchId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.TransId)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.winlost)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.winlostdate)
                .HasColumnType("smalldatetime")
                .IsRequired()
            ;

            builder.Property(x => x.status)
                .HasColumnType("nvarchar(10)")
                .IsRequired()
                .HasMaxLength(10)
            ;

            builder.Property(x => x.statuswinlost)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.betstatus)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ruben)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CommissionableStake)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.IsProcessed)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

        }
    }
}
