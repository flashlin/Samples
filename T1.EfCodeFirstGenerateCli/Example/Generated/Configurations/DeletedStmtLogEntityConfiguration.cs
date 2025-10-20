using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class DeletedStmtLogEntityConfiguration : IEntityTypeConfiguration<DeletedStmtLogEntity>
    {
        public void Configure(EntityTypeBuilder<DeletedStmtLogEntity> builder)
        {
            builder.ToTable("DeletedStmtLog");

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

            builder.Property(x => x.transdate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.status)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.winlost)
                .HasColumnType("")
            ;

            builder.Property(x => x.creator)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.winlostdate)
                .HasColumnType("smalldatetime")
            ;

            builder.Property(x => x.betfrom)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
                .HasDefaultValue("i")
            ;

            builder.Property(x => x.betcheck)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
                .HasDefaultValue("")
            ;

            builder.Property(x => x.checktime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.actualrate)
                .HasColumnType("")
            ;

            builder.Property(x => x.recommend)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.mrecommend)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.srecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.ruben)
                .HasColumnType("tinyint(3,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.bettype)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.currency)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.actual_stake)
                .HasColumnType("")
            ;

            builder.Property(x => x.transdesc)
                .HasColumnType("nvarchar(300)")
                .HasMaxLength(300)
            ;

            builder.Property(x => x.ip)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
                .HasDefaultValue("")
            ;

            builder.Property(x => x.username)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.currencystr)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.betstatus)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.creatorName)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
