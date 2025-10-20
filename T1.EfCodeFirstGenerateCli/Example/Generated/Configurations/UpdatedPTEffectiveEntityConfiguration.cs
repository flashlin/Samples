using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class UpdatedPTEffectiveEntityConfiguration : IEntityTypeConfiguration<UpdatedPTEffectiveEntity>
    {
        public void Configure(EntityTypeBuilder<UpdatedPTEffectiveEntity> builder)
        {
            builder.ToTable("UpdatedPTEffective");

            builder.HasKey(x => new { x.custid, x.typeid, x.SportID });

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.type)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.typeid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.priority)
                .HasColumnType("tinyint(3,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x._5050mktSPTMin)
                .HasColumnName("5050mktSPTMin")
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktSPTMax)
                .HasColumnName("5050mktSPTMax")
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktSTakeAll)
                .HasColumnName("5050mktSTakeAll")
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktMPTMin)
                .HasColumnName("5050mktMPTMin")
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktMPTMax)
                .HasColumnName("5050mktMPTMax")
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktMTakeAll)
                .HasColumnName("5050mktMTakeAll")
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktAPTMin)
                .HasColumnName("5050mktAPTMin")
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktAPTMax)
                .HasColumnName("5050mktAPTMax")
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktATakeAll)
                .HasColumnName("5050mktATakeAll")
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktSPT)
                .HasColumnName("5050mktSPT")
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktMPT)
                .HasColumnName("5050mktMPT")
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x._5050mktAPT)
                .HasColumnName("5050mktAPT")
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktSPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktSPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktSTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktMPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktMPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktMTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktAPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktAPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktATakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktSPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktMPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.non5050mktAPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktSPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktSPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktSTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktMPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktMPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktMTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktAPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktAPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktATakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktSPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktMPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.HDPLivemktAPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktSPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktSPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktSTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktMPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktMPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktMTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktAPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktAPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktATakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktSPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktMPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.OULivemktAPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktSPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktSPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktSTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktMPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktMPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktMTakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktAPTMin)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktAPTMax)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktATakeAll)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktSPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktMPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.ORmktAPT)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.SportID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.LastModifiedDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.IsUpdated)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

        }
    }
}
