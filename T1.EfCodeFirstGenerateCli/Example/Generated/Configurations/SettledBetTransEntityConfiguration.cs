using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SettledBetTransEntityConfiguration : IEntityTypeConfiguration<SettledBetTransEntity>
    {
        public void Configure(EntityTypeBuilder<SettledBetTransEntity> builder)
        {
            builder.ToTable("SettledBetTrans");

            builder.HasKey(x => new { x.TransId, x.MatchResultId, x.ActionId });

            builder.Property(x => x.TransId)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.MatchResultId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ActionName)
                .HasColumnType("varchar(100)")
                .IsRequired()
                .HasMaxLength(100)
            ;

            builder.Property(x => x.ActionId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
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
