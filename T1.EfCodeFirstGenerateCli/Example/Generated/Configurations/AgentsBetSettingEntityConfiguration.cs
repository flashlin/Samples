using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AgentsBetSettingEntityConfiguration : IEntityTypeConfiguration<AgentsBetSettingEntity>
    {
        public void Configure(EntityTypeBuilder<AgentsBetSettingEntity> builder)
        {
            builder.ToTable("AgentsBetSetting");

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

            builder.Property(x => x.parentid)
                .HasColumnType("int(10,0)")
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

            builder.Property(x => x.maxcredit)
                .HasColumnType("decimal(19,2)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.lastTxnDate)
                .HasColumnType("datetime")
            ;

        }
    }
}
