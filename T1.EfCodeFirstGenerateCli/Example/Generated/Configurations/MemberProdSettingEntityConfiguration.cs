using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MemberProdSettingEntityConfiguration : IEntityTypeConfiguration<MemberProdSettingEntity>
    {
        public void Configure(EntityTypeBuilder<MemberProdSettingEntity> builder)
        {
            builder.ToTable("MemberProdSetting");

            builder.HasKey(x => x.CustID);

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.AgtID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MaID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.SmaID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.HRMinDay)
                .HasColumnType("smallint(5,0)")
                .IsRequired()
                .HasDefaultValue(3)
            ;

            builder.Property(x => x.HRMaxDay)
                .HasColumnType("smallint(5,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.HREnabled)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(true)
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.GMMinDay)
                .HasColumnType("smallint(5,0)")
                .IsRequired()
                .HasDefaultValue(3)
            ;

            builder.Property(x => x.GMMaxDay)
                .HasColumnType("smallint(5,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.GMEnabled)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(true)
            ;

            builder.Property(x => x.CASEnabled)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.SBEnabled)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(true)
            ;

            builder.Property(x => x.RToteEnabled)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.RCEnabled)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.RToteFollowEng)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.LDEnabled)
                .HasColumnType("bit")
                .HasDefaultValue(true)
            ;

            builder.Property(x => x.EFootballEnabled)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(true)
            ;

        }
    }
}
