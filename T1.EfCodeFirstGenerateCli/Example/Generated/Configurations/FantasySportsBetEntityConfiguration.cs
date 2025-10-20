using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class FantasySportsBetEntityConfiguration : IEntityTypeConfiguration<FantasySportsBetEntity>
    {
        public void Configure(EntityTypeBuilder<FantasySportsBetEntity> builder)
        {
            builder.ToTable("FantasySportsBet");

            builder.HasKey(x => new { x.BetId, x.RefNo });

            builder.Property(x => x.BetId)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.RefNo)
                .HasColumnType("char(21)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasMaxLength(21)
            ;

            builder.Property(x => x.Stake)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.BetTime)
                .HasColumnType("datetime2")
                .IsRequired()
            ;

            builder.Property(x => x.Status)
                .HasColumnType("nvarchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.BetStatus)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.CreatedTime)
                .HasColumnType("datetime2")
                .IsRequired()
            ;

            builder.Property(x => x.ActualRate)
                .HasColumnType("decimal(12,8)")
            ;

            builder.Property(x => x.AgtId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.MaId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.SmaId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.AgtPT)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.MaPT)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.SmaPT)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.PlayerCommRate)
                .HasColumnType("decimal(5,4)")
            ;

            builder.Property(x => x.AgtCommRate)
                .HasColumnType("decimal(5,4)")
            ;

            builder.Property(x => x.MaCommRate)
                .HasColumnType("decimal(5,4)")
            ;

            builder.Property(x => x.SmaCommRate)
                .HasColumnType("decimal(5,4)")
            ;

            builder.Property(x => x.DirectCustId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.MemberStatus)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.WinLostDate)
                .HasColumnType("smalldatetime")
            ;

            builder.Property(x => x.Winlost)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.AgtWinlost)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.MaWinlost)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.SmaWinlost)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.PlayerComm)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.AgtComm)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.MaComm)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.SmaComm)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.currency)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.SboCurrency)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.AccountId)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.SettlementTime)
                .HasColumnType("datetime2")
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime2")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CommissionableStake)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.IsResettle)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

        }
    }
}
