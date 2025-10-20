using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CashOutBettransmEntityConfiguration : IEntityTypeConfiguration<CashOutBettransmEntity>
    {
        public void Configure(EntityTypeBuilder<CashOutBettransmEntity> builder)
        {
            builder.ToTable("CashOutBettransm");

            builder.HasKey(x => x.ID);

            builder.Property(x => x.transid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.transdate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.oddsid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.hdp1)
                .HasColumnType("decimal(12,2)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.hdp2)
                .HasColumnType("decimal(12,2)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.odds)
                .HasColumnType("decimal(12,3)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.status)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.livehomescore)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.liveawayscore)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.liveindicator)
                .HasColumnType("bit")
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.betteam)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.refno)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.comstatus)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
                .HasDefaultValue("s")
            ;

            builder.Property(x => x.winlostdate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.betcheck)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
                .HasDefaultValue("0")
            ;

            builder.Property(x => x.matchid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.matchdate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.finalodds)
                .HasColumnType("decimal(12,3)")
            ;

            builder.Property(x => x.isfinish)
                .HasColumnType("char(1)")
                .HasMaxLength(1)
                .HasDefaultValue("N")
            ;

            builder.Property(x => x.statuswinlost)
                .HasColumnType("tinyint(3,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.ruben)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.bettype)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.sportid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.MatchResultId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.NewBetType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.DisplayType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.BetTypeGroupId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.CheckTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.BetCondition)
                .HasColumnType("nvarchar(128)")
                .HasMaxLength(128)
            ;

            builder.Property(x => x.ID)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.OldStatusWinlost)
                .HasColumnType("tinyint(3,0)")
            ;

        }
    }
}
