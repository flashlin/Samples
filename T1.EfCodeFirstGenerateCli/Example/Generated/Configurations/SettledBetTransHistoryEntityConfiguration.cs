using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SettledBetTransHistoryEntityConfiguration : IEntityTypeConfiguration<SettledBetTransHistoryEntity>
    {
        public void Configure(EntityTypeBuilder<SettledBetTransHistoryEntity> builder)
        {
            builder.ToTable("SettledBetTransHistory");


            builder.Property(x => x.TransId)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MatchResultId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ActionName)
                .HasColumnType("varchar(100)")
                .IsRequired()
                .HasMaxLength(100)
            ;

            builder.Property(x => x.ActionId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ActualStake)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.Stake)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.WinLost)
                .HasColumnType("")
                .IsRequired()
            ;

            builder.Property(x => x.WinLostDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.Status)
                .HasColumnType("nvarchar(10)")
                .IsRequired()
                .HasMaxLength(10)
            ;

            builder.Property(x => x.BetStatus)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.StatusWinlost)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.IsFreeBet)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.IsCoverBet)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.OldStatusWinlost)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.OldWinLost)
                .HasColumnType("")
            ;

            builder.Property(x => x.ID)
                .HasColumnType("bigint(19,0)")
            ;

        }
    }
}
