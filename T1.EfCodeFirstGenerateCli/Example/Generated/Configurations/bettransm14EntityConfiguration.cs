using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class bettransm14EntityConfiguration : IEntityTypeConfiguration<bettransm14Entity>
    {
        public void Configure(EntityTypeBuilder<bettransm14Entity> builder)
        {
            builder.ToTable("bettransm14");


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
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
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
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.creator)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.refno)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.comstatus)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.winlostdate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.betfrom)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.betcheck)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.checktime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.oddsspread)
                .HasColumnType("decimal(12,3)")
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

            builder.Property(x => x.matchdate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.finalodds)
                .HasColumnType("")
            ;

            builder.Property(x => x.isfinish)
                .HasColumnType("char(1)")
                .HasMaxLength(1)
            ;

            builder.Property(x => x.statuswinlost)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.ruben)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.bettype)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.homeid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.awayid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.leagueid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.srecommend)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.spositiontaking)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.sdiscount)
                .HasColumnType("decimal(5,4)")
            ;

            builder.Property(x => x.sportid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.rid)
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

            builder.Property(x => x.BetCondition)
                .HasColumnType("nvarchar(128)")
                .HasMaxLength(128)
            ;

            builder.Property(x => x.id)
                .HasColumnType("bigint(19,0)")
            ;

        }
    }
}
