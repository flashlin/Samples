using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class matchEntityConfiguration : IEntityTypeConfiguration<matchEntity>
    {
        public void Configure(EntityTypeBuilder<matchEntity> builder)
        {
            builder.ToTable("match");

            builder.HasKey(x => x.matchid);

            builder.Property(x => x.matchid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.leagueid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.homeid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.awayid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.eventdate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.eventstatus)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.betstatus)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.livehomescore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.liveawayscore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.finalhomescore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.finalawayscore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.creator)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.matchcode)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.kickofftime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.closedtime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.showtime)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.hthomescore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.htawayscore)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.hometotal)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.awaytotal)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.ruben)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.multiple)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("binary(8)")
                .HasMaxLength(8)
            ;

            builder.Property(x => x.sportid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.tlive)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.livecontrolhdp)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.livecontrolou)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.channel)
                .HasColumnType("nvarchar(512)")
                .HasMaxLength(512)
            ;

            builder.Property(x => x.result)
                .HasColumnType("nvarchar(200)")
                .HasMaxLength(200)
            ;

            builder.Property(x => x.color)
                .HasColumnType("nvarchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.remark)
                .HasColumnType("nvarchar(1000)")
                .HasMaxLength(1000)
            ;

            builder.Property(x => x.otherstatus)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.otherstatus2)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.absKickoffTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.EventType)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.EventTypeID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LiveHdpTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LiveOuTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.HdpTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.OuTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ft1x2Trader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.fh1x2Trader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ShowTimeDisplayType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LiveFhHdpTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LiveFhOuTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.CreateOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.HomeJersey)
                .HasColumnType("nvarchar(1)")
                .HasMaxLength(1)
            ;

            builder.Property(x => x.AwayJersey)
                .HasColumnType("nvarchar(1)")
                .HasMaxLength(1)
            ;

            builder.Property(x => x.ParentId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ScoreVerified)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.LiveCsTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LiveFhCsTrader)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.OwnTstamp)
                .HasColumnType("timestamp")
                .IsRequired()
            ;

        }
    }
}
