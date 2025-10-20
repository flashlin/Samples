using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SettlementTimeLogEntityConfiguration : IEntityTypeConfiguration<SettlementTimeLogEntity>
    {
        public void Configure(EntityTypeBuilder<SettlementTimeLogEntity> builder)
        {
            builder.ToTable("SettlementTimeLog");


            builder.Property(x => x.MatchResultId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.SPName)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.BetCount)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.StartTime)
                .HasColumnType("datetime2")
            ;

            builder.Property(x => x.AfterBetTrans)
                .HasColumnType("datetime2")
            ;

            builder.Property(x => x.AfterCashSettled)
                .HasColumnType("datetime2")
            ;

            builder.Property(x => x.AfterDailyStatement)
                .HasColumnType("datetime2")
            ;

            builder.Property(x => x.AfterSettledBetTrans)
                .HasColumnType("datetime2")
            ;

        }
    }
}
