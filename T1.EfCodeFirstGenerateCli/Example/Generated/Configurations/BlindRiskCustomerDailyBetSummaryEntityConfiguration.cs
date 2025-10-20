using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BlindRiskCustomerDailyBetSummaryEntityConfiguration : IEntityTypeConfiguration<BlindRiskCustomerDailyBetSummaryEntity>
    {
        public void Configure(EntityTypeBuilder<BlindRiskCustomerDailyBetSummaryEntity> builder)
        {
            builder.ToTable("BlindRiskCustomerDailyBetSummary");


            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.WinlostDate)
                .HasColumnType("smalldatetime")
                .IsRequired()
            ;

            builder.Property(x => x.Turnover)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.StakeLessThan100BetCount)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.TotalBetCount)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

        }
    }
}
