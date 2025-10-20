using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CashOutActionLogEntityConfiguration : IEntityTypeConfiguration<CashOutActionLogEntity>
    {
        public void Configure(EntityTypeBuilder<CashOutActionLogEntity> builder)
        {
            builder.ToTable("CashOutActionLog");

            builder.HasKey(x => x.LogId);

            builder.Property(x => x.LogId)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.transid)
                .HasColumnType("bigint(19,0)")
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

            builder.Property(x => x.status)
                .HasColumnType("nvarchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.winlost)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.awinlost)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.mwinlost)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.swinlost)
                .HasColumnType("decimal(19,6)")
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

            builder.Property(x => x.scomm)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.winlostdate)
                .HasColumnType("smalldatetime")
            ;

            builder.Property(x => x.statuswinlost)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.betstatus)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.MemberStatus)
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

            builder.Property(x => x.ActionType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.LogDate)
                .HasColumnType("datetime")
            ;

        }
    }
}
