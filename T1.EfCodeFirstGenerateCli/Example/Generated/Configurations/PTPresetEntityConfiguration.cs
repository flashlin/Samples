using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class PTPresetEntityConfiguration : IEntityTypeConfiguration<PTPresetEntity>
    {
        public void Configure(EntityTypeBuilder<PTPresetEntity> builder)
        {
            builder.ToTable("PTPreset");

            builder.HasKey(x => x.custid);

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.roleid)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.parentid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.type)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.typeid)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(1)
            ;

            builder.Property(x => x.priority)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x._5050mktPTMin)
                .HasColumnName("5050mktPTMin")
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktPTMax)
                .HasColumnName("5050mktPTMax")
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktTakeAll)
                .HasColumnName("5050mktTakeAll")
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedDate)
                .HasColumnType("datetime")
            ;

        }
    }
}
