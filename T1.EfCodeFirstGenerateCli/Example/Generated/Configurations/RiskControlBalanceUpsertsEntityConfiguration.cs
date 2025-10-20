using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class RiskControlBalanceUpsertsEntityConfiguration : IEntityTypeConfiguration<RiskControlBalanceUpsertsEntity>
    {
        public void Configure(EntityTypeBuilder<RiskControlBalanceUpsertsEntity> builder)
        {
            builder.ToTable("RiskControlBalanceUpserts");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.WinLose)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.StartingRiskBalance)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.TryReset)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.RequestTime)
                .HasColumnType("datetime")
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
