using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AutoCBSettingEntityConfiguration : IEntityTypeConfiguration<AutoCBSettingEntity>
    {
        public void Configure(EntityTypeBuilder<AutoCBSettingEntity> builder)
        {
            builder.ToTable("AutoCBSetting");

            builder.HasKey(x => x.ID);

            builder.Property(x => x.ID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ProductType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.SubProductType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.isIOM)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.FromID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.FromAccount)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ToID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ToAccount)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.TxnType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.Description)
                .HasColumnType("varchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.IsTest)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.AutoCBType)
                .HasColumnType("int(10,0)")
            ;

        }
    }
}
