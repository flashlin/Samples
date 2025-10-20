using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MemberBetSettingEntityConfiguration : IEntityTypeConfiguration<MemberBetSettingEntity>
    {
        public void Configure(EntityTypeBuilder<MemberBetSettingEntity> builder)
        {
            builder.ToTable("MemberBetSetting");

            builder.HasKey(x => new { x.custid, x.sportid, x.bettype });

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.sportid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.bettype)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.recommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.minbet)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.maxbet)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.maxpermatch)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.remark)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.modifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.modifiedDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.credit)
                .HasColumnType("decimal(19,2)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.lastBetDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.lastTxnDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.lastWinLostDate)
                .HasColumnType("smalldatetime")
            ;

        }
    }
}
