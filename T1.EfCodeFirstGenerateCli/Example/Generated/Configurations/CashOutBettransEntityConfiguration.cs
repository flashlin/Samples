using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CashOutBettransEntityConfiguration : IEntityTypeConfiguration<CashOutBettransEntity>
    {
        public void Configure(EntityTypeBuilder<CashOutBettransEntity> builder)
        {
            builder.ToTable("CashOutBettrans");

            builder.HasKey(x => x.transid);

            builder.Property(x => x.transid)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.refno)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.username)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.transdate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.oddsid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.hdp1)
                .HasColumnType("decimal(12,2)")
            ;

            builder.Property(x => x.hdp2)
                .HasColumnType("decimal(12,2)")
            ;

            builder.Property(x => x.odds)
                .HasColumnType("decimal(12,3)")
            ;

            builder.Property(x => x.stake)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.status)
                .HasColumnType("nvarchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.winlost)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.livehomescore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.liveawayscore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.liveindicator)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.betteam)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.winlostdate)
                .HasColumnType("smalldatetime")
            ;

            builder.Property(x => x.betfrom)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.oddsspread)
                .HasColumnType("")
            ;

            builder.Property(x => x.apositiontaking)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.mpositiontaking)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.tpositiontaking)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.awinlost)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.mwinlost)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.playerdiscount)
                .HasColumnType("decimal(5,4)")
            ;

            builder.Property(x => x.discount)
                .HasColumnType("decimal(5,4)")
            ;

            builder.Property(x => x.adiscount)
                .HasColumnType("decimal(5,4)")
            ;

            builder.Property(x => x.playercomm)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.comm)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.acomm)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.actualrate)
                .HasColumnType("decimal(12,8)")
            ;

            builder.Property(x => x.matchid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.recommend)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.mrecommend)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ruben)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.statuswinlost)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.bettype)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.currency)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.actual_stake)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.sdiscount)
                .HasColumnType("decimal(5,4)")
            ;

            builder.Property(x => x.scomm)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.spositiontaking)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.swinlost)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.srecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.currencystr)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.oddsstyle)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.betstatus)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.sportid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.leagueid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.DangerLevel)
                .HasColumnType("tinyint(3,0)")
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

            builder.Property(x => x.BetCondition)
                .HasColumnType("nvarchar(128)")
                .HasMaxLength(128)
            ;

            builder.Property(x => x.BetTypeGroupId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CashOutTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CashOutValue)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.MemberStatus)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.CashOutScomm)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.SettlementTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.JackTimeTransactionId)
                .HasColumnType("varchar(150)")
                .HasMaxLength(150)
            ;

            builder.Property(x => x.JackTimeRemark)
                .HasColumnType("varchar(150)")
                .HasMaxLength(150)
            ;

            builder.Property(x => x.modds)
                .HasColumnType("")
            ;

        }
    }
}
