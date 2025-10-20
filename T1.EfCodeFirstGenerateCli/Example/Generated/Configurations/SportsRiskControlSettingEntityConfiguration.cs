using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SportsRiskControlSettingEntityConfiguration : IEntityTypeConfiguration<SportsRiskControlSettingEntity>
    {
        public void Configure(EntityTypeBuilder<SportsRiskControlSettingEntity> builder)
        {
            builder.ToTable("SportsRiskControlSetting");

            builder.HasKey(x => x.CustomerId);

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.MaxWin)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MaxLose)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.IsMaxWinUnlimited)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.IsMaxLoseUnlimited)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.ResetProfileId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

        }
    }
}
