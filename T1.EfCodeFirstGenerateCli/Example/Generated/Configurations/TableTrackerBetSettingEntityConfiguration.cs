using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class TableTrackerBetSettingEntityConfiguration : IEntityTypeConfiguration<TableTrackerBetSettingEntity>
    {
        public void Configure(EntityTypeBuilder<TableTrackerBetSettingEntity> builder)
        {
            builder.ToTable("TableTrackerBetSetting");

            builder.HasKey(x => x.rid);

            builder.Property(x => x.rid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.SportId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.BetType)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.TableName)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
